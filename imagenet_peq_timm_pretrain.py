"""ImageNet-1K PEQ pretraining: 800 epochs with the ViT-5 data recipe."""

from peq_timm_common import create_stage_parser, run_stage


def main() -> None:
    args = create_stage_parser("pretrain").parse_args()
    run_stage(args, "pretrain")


if __name__ == "__main__":
    main()
