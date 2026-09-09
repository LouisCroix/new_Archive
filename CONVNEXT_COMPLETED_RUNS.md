# Completed ConvNeXt Run Summary

## Scope and conventions

This report covers every ConvNeXt-related run under `outputs/` that is complete at the time of the audit. A run is considered complete only when:

1. its recorded epoch count equals its configured epoch count; and
2. `checkpoint_final.pt` exists.

There are **11 completed runs**. Five incomplete or merely configured runs are excluded; they are listed at the end of this report.

All Top-1 values below are percentages. For the standard trainer, **Best Val Top-1** is the maximum `val_acc1` in `history.json`. For the official-recipe trainer, it is the maximum raw-model validation `acc1` in `metrics.jsonl`. The official run also reports EMA accuracy separately because its `checkpoint_best.pt` selection is based on EMA Top-1. Epoch indices are the zero-based values stored in the logs.

`ARR1` gives the number of unique blocks per stage, while `ARR2` gives the number of shared-parameter applications of each complete stage. V1 denotes the torchvision-style ConvNeXt block with LayerScale; V2 denotes the ConvNeXt V2 block with GRN.

## Completed-run inventory

| ID | Output directory | Architecture hyperparameters | Parameters | Training hyperparameters | Best Val Top-1 | Best epoch |
|---|---|---|---:|---|---:|---:|
| R01 | `imagenet_recurrent_v2_convnext_T12_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | Legacy v2 naive ConvNeXt V1; one shared 96-wide block; `T=12`; equivalent `ARR1=1,0,0,0`, `ARR2=12,0,0,0` | 0.181M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **36.584%** | 20 |
| R02 | `imagenet_recurrent_v3_pro_convnext_depth1_T12_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | Legacy v3 Pro, ConvNeXt V1; `depth=1`, `T=12`; equivalent `ARR1=3,3,1,0`, `ARR2=1,1,12,0`; 18 block applications | 3.118M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **67.156%** | 21 |
| R03 | `imagenet_recurrent_v4_promax_convnext_depth1_T12_img224_epochs22_BS32_accum4_lr5e-4_minlr1e-6` | Legacy v4 Promax, ConvNeXt V1; `depth=1`, `T=12`; equivalent `ARR1=3,3,1,0`, `ARR2=12,12,12,0`; 84 block applications | 3.118M | 22 epochs; warmup 2; batch 32; accumulation 4; LR `5e-4 -> 1e-6` | **70.540%** | 21 |
| R04 | `imagenet_recurrent_v6_convnext_ARR1-1-1-1-0_ARR2-3-3-9-0_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | v6 ConvNeXt V1; `ARR1=1,1,1,0`; `ARR2=3,3,9,0`; 3 unique / 15 applied blocks; 3 stages; last width 384 | 2.348M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **65.810%** | 21 |
| R05 | `imagenet_recurrent_v6_convnext_ARR1-1-1-1-1_ARR2-3-3-9-3_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | v6 ConvNeXt V1; `ARR1=1,1,1,1`; `ARR2=3,3,9,3`; 4 unique / 18 applied blocks; 4 stages; last width 768 | 8.677M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **69.554%** | 21 |
| R06 | `imagenet_recurrent_v6_convnext_ARR1-1-1-2-0_ARR2-3-3-6-0_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | v6 ConvNeXt V1; `ARR1=1,1,2,0`; `ARR2=3,3,6,0`; 4 unique / 18 applied blocks; 3 stages; last width 384 | 3.550M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **67.928%** | 21 |
| R07 | `imagenet_recurrent_v6_convnextV2_ARR1-1-1-2-0_ARR2-3-3-6-0_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | v6 ConvNeXt V2; `ARR1=1,1,2,0`; `ARR2=3,3,6,0`; 4 unique / 18 applied blocks; 3 stages; last width 384 | 3.557M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **70.326%** | 21 |
| R08 | `imagenet_recurrent_v6_convnextV2_ARR1-1-1-2-0_ARR2-3-3-6-0_img224_epochs100_BS128_accum1_lr5e-4_minlr1e-6` | Same architecture as R07 | 3.557M | 100 epochs; warmup 5; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **74.384%** | 98 |
| R09 | `imagenet_recurrent_v7_convnextV2_ARR1-1-1-1-0_ARR2-3-3-6-0_REG-0-0-1-0_NREG-8-8-64-8_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | v7 ConvNeXt V2 + RATS registers; `ARR1=1,1,1,0`; `ARR2=3,3,6,0`; `REG=0,0,1,0`; 64 stage-3 registers; 6 attention heads; 3 unique / 12 applied ConvNeXt blocks; feature-mean readout | 4.449M | 22 epochs; warmup 2; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **70.552%** | 21 |
| R10 | `imagenet_recurrent_v7_convnextV2_ARR1-1-1-1-0_ARR2-3-3-6-0_REG-0-0-1-0_NREG-8-8-64-8_img224_epochs100_BS128_accum1_lr5e-4_minlr1e-6` | Same architecture as R09 | 4.449M | 100 epochs; warmup 5; batch 128; accumulation 1; LR `5e-4 -> 1e-6` | **75.650%** | 98 |
| R11 | `imagenet_recurrent_official_convnextV2_ARR1-1-1-2-0_ARR2-3-3-6-0_ep100_warmup5_gbs4096_seed0` | Official-trainer ConvNeXt V2; `ARR1=1,1,2,0`; `ARR2=3,3,6,0`; 4 unique / 18 applied blocks; drop-path 0.1 with an unrolled linear schedule | 3.557M | 100 epochs; warmup 5; 4 GPUs x 128/GPU x accumulation 8 = global batch 4096; peak/min LR `4e-3/1e-6`; Mixup 0.8; CutMix 1.0; RandAugment; random erasing 0.25; EMA 0.9999 | **74.306% raw**; 68.026% EMA | 96 raw; 99 EMA |

### Common standard-trainer settings

R01-R10 use the full ImageNet-1K training and validation sets, 224-pixel input with resize 256, AdamW, weight decay 0.05, no decay on one-dimensional parameters and biases, label smoothing 0.1, BF16 AMP, and seed 0. Their batch and accumulation values are the values recorded in each run's configuration. R06-R10 explicitly record stochastic depth as 0.0.

R11 also uses ImageNet-1K, AdamW, weight decay 0.05, label smoothing 0.1, BF16 AMP, and seed 0. It is labelled `training_recipe_exact=false` in `config.json`: it borrows the nominal 300-epoch ConvNeXt recipe but changes the schedule to 100 epochs with 5 warmup epochs.

## Comparisons with clear experimental relevance

### 1. ConvNeXt V1 versus V2 block: controlled 22-epoch comparison

R06 and R07 have the same stage arrays, number of block applications, schedule, batch size, optimizer settings, and seed. The primary architecture change is the ConvNeXt block version.

| Run | Block | ARR1 / ARR2 | Params | Best Val Top-1 | Difference |
|---|---|---|---:|---:|---:|
| R06 | ConvNeXt V1 / LayerScale | `1,1,2,0` / `3,3,6,0` | 3.550M | 67.928% | baseline |
| R07 | ConvNeXt V2 / GRN | `1,1,2,0` / `3,3,6,0` | 3.557M | 70.326% | **+2.398 pp** |

Under this setup, replacing the V1 block with the V2/GRN block improves Best Val Top-1 by **2.398 percentage points**.

### 2. Training duration: exact within-architecture comparisons

| Architecture | 22 epochs | 100 epochs | Gain from longer training |
|---|---:|---:|---:|
| v6 ConvNeXt V2, `ARR1=1,1,2,0`, `ARR2=3,3,6,0` (R07 -> R08) | 70.326% | 74.384% | **+4.058 pp** |
| v7 ConvNeXt V2 + stage-3 RATS registers (R09 -> R10) | 70.552% | 75.650% | **+5.098 pp** |

The warmup also changes from 2 epochs to 5 epochs, so these values represent the complete 22-epoch versus 100-epoch schedules rather than epoch count alone.

### 3. RATS-register design versus the non-register baseline

| Schedule | Non-register v6 | RATS-register v7 | Difference |
|---|---:|---:|---:|
| 22 epochs (R07 vs R09) | 70.326% | 70.552% | **+0.226 pp** |
| 100 epochs (R08 vs R10) | 74.384% | 75.650% | **+1.266 pp** |

This is a meaningful design-level comparison, but it is **not a single-variable ablation**. The RATS runs add 64 registers at stage 3 and change stage-3 `ARR1` from 2 to 1, reducing ConvNeXt block applications from 18 to 12 while increasing parameters from 3.557M to 4.449M.

### 4. Standard trainer versus official-style recipe at 100 epochs

R08 and R11 use the same ConvNeXt V2 stage arrays and parameter count, but substantially different optimization and augmentation recipes.

| Run | Global/effective batch | Peak LR | Drop path | Extra augmentation / EMA | Best raw Val Top-1 |
|---|---:|---:|---:|---|---:|
| R08, standard trainer | recorded batch 128, accumulation 1 | `5e-4` | 0.0 | none recorded beyond the standard pipeline | **74.384%** |
| R11, official-style trainer | 4096 | `4e-3` | 0.1 | Mixup, CutMix, RandAugment, random erasing, EMA | **74.306%** |

The raw-model difference is **-0.078 pp** for the official-style run. Because many training variables change simultaneously, this near tie should be treated as a recipe-level result, not evidence that the individual additions have no effect. R11's best EMA Top-1 is 68.026%, and should not be substituted for its raw-model result when comparing with R08.

### 5. ConvNeXt V1 stage-allocation comparisons at 22 epochs

| Run | ARR1 / ARR2 | Active stages | Applied blocks | Params | Best Val Top-1 | Difference vs R04 |
|---|---|---:|---:|---:|---:|---:|
| R04 | `1,1,1,0` / `3,3,9,0` | 3 | 15 | 2.348M | 65.810% | baseline |
| R06 | `1,1,2,0` / `3,3,6,0` | 3 | 18 | 3.550M | 67.928% | **+2.118 pp** |
| R05 | `1,1,1,1` / `3,3,9,3` | 4 | 18 | 8.677M | 69.554% | **+3.744 pp** |

These runs share the same standard 22-epoch training settings, but they change stage allocation, width reached by the network, and parameter count. They are therefore useful architecture comparisons rather than isolated ablations.

### 6. Legacy Pro versus Promax recurrence placement

| Run | Mode | Equivalent ARR1 / ARR2 | Applied blocks | Params | Recorded batch x accumulation | Best Val Top-1 |
|---|---|---|---:|---:|---:|---:|
| R02 | Pro | `3,3,1,0` / `1,1,12,0` | 18 | 3.118M | 128 x 1 | 67.156% |
| R03 | Promax | `3,3,1,0` / `12,12,12,0` | 84 | 3.118M | 32 x 4 | 70.540% |

Promax is **+3.384 pp** above Pro at the same parameter count. It applies tied recurrence in all three active stages, so the improvement comes with much more computation. The microbatch and accumulation configuration also differs even though both products equal 128.

## Overall ranking

| Rank | Run | Best Val Top-1 |
|---:|---|---:|
| 1 | R10: ConvNeXt V2 + RATS registers, 100 epochs | **75.650%** |
| 2 | R08: ConvNeXt V2, standard trainer, 100 epochs | **74.384%** |
| 3 | R11: ConvNeXt V2, official-style trainer, 100 epochs | **74.306% raw** |
| 4 | R09: ConvNeXt V2 + RATS registers, 22 epochs | **70.552%** |
| 5 | R03: legacy Promax ConvNeXt V1, 22 epochs | **70.540%** |
| 6 | R07: ConvNeXt V2, 22 epochs | **70.326%** |
| 7 | R05: four-stage ConvNeXt V1, 22 epochs | **69.554%** |
| 8 | R06: three-stage ConvNeXt V1 with two unique stage-3 blocks, 22 epochs | **67.928%** |
| 9 | R02: legacy Pro ConvNeXt V1, 22 epochs | **67.156%** |
| 10 | R04: three-stage ConvNeXt V1 with one unique block per stage, 22 epochs | **65.810%** |
| 11 | R01: one-block naive ConvNeXt V1, 22 epochs | **36.584%** |

## Excluded incomplete runs

The following ConvNeXt-related directories were not included in the completed-run tables:

| Output directory | Recorded progress | Reason for exclusion |
|---|---:|---|
| `imagenet_recurrent_v5_promini_convnext_depth1_T12_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | 20 / 22 epochs | No final checkpoint |
| `imagenet_recurrent_v6_convnextV2_ARR1-3-3-2-0_ARR2-1-1-6-0_img224_epochs22_BS128_accum1_lr5e-4_minlr1e-6` | 1 / 22 epochs | No final checkpoint |
| `imagenet_recurrent_official_convnextV2_ARR1-1-1-2-0_ARR2-3-3-6-0_ep100_warmup5_gbs512_maxlr0.0005_seed0` | 32 / 100 epochs | No final checkpoint |
| `imagenet_recurrent_official_convnextV2_ARR1-1-1-2-0_ARR2-3-3-6-0_ep300_warmup20_gbs512_maxlr0.0005_seed0` | 0 / 300 epochs | Configuration only |
| `imagenet_recurrent_v8_convnextV2_ARR1-1-1-1-0_ARR2-3-3-6-0_REG-0-0-1-0_NREG-8-8-64-8_DELTA1_REGHEAD1_img224_epochs100_BS64_accum2_lr5e-4_minlr1e-6` | 6 / 100 epochs | No final checkpoint |
