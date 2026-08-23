"""ImageNet-1K PEQ fine-tuning from a pretraining EMA checkpoint."""

from peq_timm_common import create_stage_parser, run_stage


def main() -> None:
    args = create_stage_parser("finetune").parse_args()
    run_stage(args, "finetune")


if __name__ == "__main__":
    main()
