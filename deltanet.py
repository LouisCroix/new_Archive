"""Delta attention modules with readable and official FLA execution backends."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


DELTA_BACKENDS = {"auto", "naive", "fla", "chunk", "fused_recurrent"}
_FLA_OPS = None
_FLA_SHORT_CONV = None
_FLA_IMPORT_ERROR = None
_FLA_FALLBACK_WARNED = False


def _load_fla_ops():
    global _FLA_OPS, _FLA_IMPORT_ERROR
    if _FLA_OPS is not None:
        return _FLA_OPS
    if _FLA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The official FLA DeltaNet backend is unavailable. Install fla-core "
            "and einops in the training environment."
        ) from _FLA_IMPORT_ERROR
    try:
        from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
    except (ImportError, OSError) as exc:
        _FLA_IMPORT_ERROR = exc
        raise RuntimeError(
            "The official FLA DeltaNet backend is unavailable. Install fla-core "
            "and einops in the training environment."
        ) from exc
    _FLA_OPS = (chunk_delta_rule, fused_recurrent_delta_rule)
    return _FLA_OPS


def require_fla():
    """Fail early when an explicitly requested FLA backend cannot be imported."""
    _load_fla_ops()


def _load_fla_short_conv():
    global _FLA_SHORT_CONV
    if _FLA_SHORT_CONV is None:
        _load_fla_ops()
        try:
            from fla.modules import ShortConvolution
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "The installed FLA package does not provide ShortConvolution. "
                "Install the versions listed in requirements-delta.txt."
            ) from exc
        _FLA_SHORT_CONV = ShortConvolution
    return _FLA_SHORT_CONV


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight.to(dtype)


class CausalDepthwiseConv1d(nn.Module):
    """Small causal convolution used after the Q/K/V projections."""

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            groups=dim,
            bias=False,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)


class MultiheadMixin:
    def __init__(self, dim, heads):
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    def _h(self, x):
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.heads, self.head_dim).transpose(1, 2)

    def _m(self, x):
        bsz, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq, self.dim)

    def _validate_inputs(self, query, key, value):
        if (key is None) != (value is None):
            raise ValueError("key and value must either both be provided or both be omitted")
        tensors = (query,) if key is None else (query, key, value)
        if any(x.ndim != 3 for x in tensors):
            raise ValueError("attention inputs must have shape [batch, sequence, dim]")
        if any(x.shape[0] != query.shape[0] for x in tensors):
            raise ValueError("query, key, and value must have the same batch size")
        if any(x.shape[-1] != self.dim for x in tensors):
            raise ValueError(f"attention input dim must be {self.dim}")
        if key is not None and key.shape[1] != value.shape[1]:
            raise ValueError("key and value must have the same sequence length")
        return key is not None


def _delta_update(state, key, value, beta):
    old_value = torch.einsum("bhde,bhd->bhe", state, key)
    prediction_error = value - old_value
    return state + beta[..., None, None] * torch.einsum(
        "bhd,bhe->bhde",
        key,
        prediction_error,
    )


def _cross_delta_read(query, key, value, beta):
    state = key.new_zeros(
        key.shape[0],
        key.shape[1],
        key.shape[-1],
        value.shape[-1],
    )
    for i in range(key.shape[2]):
        state = _delta_update(state, key[:, :, i], value[:, :, i], beta[:, :, i])
    return torch.einsum("bhde,bhld->bhle", state, query)


def _naive_delta_output(query, key, value, beta, cross_attention):
    if cross_attention:
        return _cross_delta_read(query, key, value, beta)

    state = key.new_zeros(
        key.shape[0],
        key.shape[1],
        key.shape[-1],
        value.shape[-1],
    )
    output = []
    for i in range(key.shape[2]):
        state = _delta_update(state, key[:, :, i], value[:, :, i], beta[:, :, i])
        output.append(
            torch.einsum("bhde,bhd->bhe", state, query[:, :, i])
        )
    return torch.stack(output, dim=2)


def _resolve_backend(backend, key):
    global _FLA_FALLBACK_WARNED
    if backend not in DELTA_BACKENDS:
        options = ", ".join(sorted(DELTA_BACKENDS))
        raise ValueError(f"Unsupported Delta backend={backend}; use {options}")
    if backend == "naive":
        return "naive"
    if backend == "auto":
        if not key.is_cuda or key.dtype == torch.float32:
            return "naive"
        try:
            _load_fla_ops()
        except RuntimeError:
            if not _FLA_FALLBACK_WARNED:
                warnings.warn(
                    "FLA is unavailable; DELTA_BACKEND=auto is using the memory-heavy "
                    "naive recurrence.",
                    stacklevel=3,
                )
                _FLA_FALLBACK_WARNED = True
            return "naive"
        return "fused_recurrent" if key.shape[2] <= 64 else "chunk"
    if not key.is_cuda:
        raise RuntimeError(f"DELTA_BACKEND={backend} requires CUDA tensors")
    if backend in {"fla", "chunk"} and key.dtype == torch.float32:
        raise RuntimeError(
            f"DELTA_BACKEND={backend} requires AMP float16/bfloat16 inputs"
        )
    if backend == "fla":
        return "fused_recurrent" if key.shape[2] <= 64 else "chunk"
    return backend


def _fla_delta_output(query, key, value, beta, cross_attention, backend, chunk_size):
    chunk_delta_rule, fused_recurrent_delta_rule = _load_fla_ops()
    op = chunk_delta_rule if backend == "chunk" else fused_recurrent_delta_rule

    if not (query.dtype == key.dtype == value.dtype == beta.dtype):
        raise RuntimeError(
            "FLA DeltaNet inputs must share one dtype; got "
            f"q={query.dtype}, k={key.dtype}, v={value.dtype}, beta={beta.dtype}"
        )
    if backend == "chunk" and query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            f"FLA chunk DeltaNet requires float16/bfloat16 inputs, got {query.dtype}"
        )

    # FLA uses [B, T, H, D]. Inputs are already normalized exactly as in the
    # readable implementation, and scale=1 keeps the existing state @ query rule.
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    b = beta.transpose(1, 2).contiguous()
    kwargs = {
        "beta": b,
        "scale": 1.0,
        "output_final_state": cross_attention,
        "use_qk_l2norm_in_kernel": False,
    }
    if backend == "chunk":
        kwargs["chunk_size"] = chunk_size

    if not cross_attention:
        output, _ = op(q=q, k=k, v=v, **kwargs)
        return output.transpose(1, 2).contiguous()

    # The official op expects aligned Q/K/V lengths. Cross attention only needs
    # the source's final memory, so K is a harmless dummy query and its output is
    # connected with zero weight to ensure the custom backward receives `do`.
    dummy_output, final_state = op(q=k, k=k, v=v, **kwargs)
    final_state = final_state.to(query.dtype)
    output = torch.einsum("bhde,bhld->bhle", final_state, query)
    return output + dummy_output[0, 0, 0, 0].to(output.dtype) * 0.0


def _run_delta_rule(
    query,
    key,
    value,
    beta,
    cross_attention,
    backend,
    chunk_size,
):
    resolved = _resolve_backend(backend, key)
    if resolved == "naive":
        return _naive_delta_output(query, key, value, beta, cross_attention)
    return _fla_delta_output(
        query,
        key,
        value,
        beta,
        cross_attention,
        resolved,
        chunk_size,
    )


def _attention_compute_dtype(reference):
    """Return the low-precision dtype selected by CUDA autocast, when active."""
    if reference.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        if dtype in {torch.float16, torch.bfloat16}:
            return dtype
    return reference.dtype


class HybridAttn(nn.Module, MultiheadMixin):
    """One QKV projection; softmax and delta reads mixed by a per-head gate."""

    def __init__(self, dim, heads, backend="auto", chunk_size=64):
        nn.Module.__init__(self)
        MultiheadMixin.__init__(self, dim, heads)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.backend = backend
        self.chunk_size = chunk_size

        # Hybrid difference: this beta projection has a learnable bias.
        self.bt = nn.Linear(dim, heads)

        # Hybrid difference: standard DeltaNet has no softmax/delta mixing gate.
        self.g = nn.Parameter(torch.full((heads,), 4.0))

        # Hybrid difference: zero initialization makes the surrounding residual
        # block start as identity; standard DeltaNet uses normal initialization.
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, query, key=None, value=None):
        cross_attention = self._validate_inputs(query, key, value)
        source = key if cross_attention else query
        source_value = value if cross_attention else query
        q_projection = self.q(query)
        compute_dtype = _attention_compute_dtype(q_projection)
        q = self._h(q_projection).to(compute_dtype)

        # Hybrid difference: only K is normalized. Standard DeltaNet applies a
        # SiLU feature map and L2 normalization to both Q and K.
        k = F.normalize(self._h(self.k(source)), dim=-1).to(compute_dtype)
        v = self._h(self.v(source_value)).to(compute_dtype)

        # Hybrid difference: self mode uses bidirectional all-pairs attention;
        # cross mode uses true query-to-source attention. The Delta branch scans
        # source tokens in their sequence order in both cases.
        o_soft = F.scaled_dot_product_attention(q, k, v)

        beta = self.bt(source).sigmoid().transpose(1, 2).to(compute_dtype)
        o_delta = _run_delta_rule(
            q, k, v, beta, cross_attention, self.backend, self.chunk_size
        )

        g = self.g.sigmoid().to(o_delta.dtype)[None, :, None, None]
        return self.o(self._m(g * o_delta + (1.0 - g) * o_soft))


class DeltaNet(nn.Module, MultiheadMixin):
    """Standard DeltaNet with interchangeable exact delta-rule backends.

    Projection and readout formulas stay local so PEQ cross attention remains
    explicit; the state recurrence can use the readable scan or official FLA.
    """

    def __init__(
        self,
        dim,
        heads,
        conv_size=4,
        norm_eps=1e-5,
        backend="auto",
        chunk_size=64,
    ):
        nn.Module.__init__(self)
        MultiheadMixin.__init__(self, dim, heads)
        self.backend = backend
        self.chunk_size = chunk_size
        self.uses_fla_conv = backend in {"fla", "chunk", "fused_recurrent"}

        # Standard DeltaNet difference: projection layers are bias-free.
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        # Standard DeltaNet difference: beta is also bias-free. It remains one
        # sigmoid scalar per token and head, exactly as in the Hybrid branch.
        self.bt = nn.Linear(dim, heads, bias=False)

        # Standard DeltaNet difference: short causal convolutions inject local
        # sequence information before the recurrent associative-memory scan.
        if self.uses_fla_conv:
            short_conv = _load_fla_short_conv()
            make_conv = lambda: short_conv(
                hidden_size=dim,
                kernel_size=conv_size,
                bias=False,
                activation="silu",
            )
        else:
            make_conv = lambda: CausalDepthwiseConv1d(dim, conv_size)
        self.q_conv = make_conv()
        self.k_conv = make_conv()
        self.v_conv = make_conv()

        # Standard DeltaNet difference: normalize every head before the final
        # output projection. The same RMSNorm parameters are shared by heads.
        self.out_norm = RMSNorm(self.head_dim, norm_eps)

    def _project(self, query, key=None, value=None):
        cross_attention = key is not None
        source = key if cross_attention else query
        source_value = value if cross_attention else query

        # Standard DeltaNet uses short-conv + SiLU for all Q/K/V features.
        def short_conv(module, x):
            output = module(x)
            if self.uses_fla_conv:
                return output[0]
            return F.silu(output)

        q_projection = self.q(query)
        compute_dtype = _attention_compute_dtype(q_projection)
        q = short_conv(self.q_conv, q_projection)
        k = short_conv(self.k_conv, self.k(source))
        v = short_conv(self.v_conv, self.v(source_value))

        # Standard DeltaNet L2-normalizes both Q and K.
        q = F.normalize(self._h(q), dim=-1).to(compute_dtype)
        k = F.normalize(self._h(k), dim=-1).to(compute_dtype)
        v = self._h(v).to(compute_dtype)
        beta = self.bt(source).sigmoid().transpose(1, 2).to(compute_dtype)
        return q, k, v, beta

    def _delta_output(self, q, k, v, beta, cross_attention):
        # Standard delta rule:
        # S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1} k_t)^T.
        return _run_delta_rule(
            q, k, v, beta, cross_attention, self.backend, self.chunk_size
        )

    def forward(self, query, key=None, value=None):
        cross_attention = self._validate_inputs(query, key, value)
        q, k, v, beta = self._project(query, key, value)
        output = self._delta_output(q, k, v, beta, cross_attention)

        # Standard DeltaNet difference: pure delta read, with no softmax branch
        # and no learned mixture between two attention mechanisms.
        return self.o(self._m(self.out_norm(output)))


class DeltaHybridAttn(DeltaNet):
    """Standard DeltaNet features mixed with scaled dot-product attention."""

    def __init__(
        self,
        dim,
        heads,
        conv_size=4,
        norm_eps=1e-5,
        backend="auto",
        chunk_size=64,
    ):
        super().__init__(dim, heads, conv_size, norm_eps, backend, chunk_size)

        # Delbrid difference: standard DeltaNet has no softmax branch or gate.
        # The initialization matches HybridAttn and initially favors delta reads.
        self.g = nn.Parameter(torch.full((heads,), 4.0))

    def forward(self, query, key=None, value=None):
        cross_attention = self._validate_inputs(query, key, value)
        q, k, v, beta = self._project(query, key, value)

        o_delta = self._delta_output(q, k, v, beta, cross_attention)
        # Normalize only the DeltaNet branch so g -> 1 recovers standard DeltaNet.
        o_delta = self.out_norm(o_delta)

        # Delbrid difference: add a shared-feature SDPA branch. In cross mode this
        # is true query-to-source attention; in self mode it is bidirectional.
        o_soft = F.scaled_dot_product_attention(q, k, v)
        g = self.g.sigmoid().to(o_delta.dtype)[None, :, None, None]
        return self.o(self._m(g * o_delta + (1.0 - g) * o_soft))
