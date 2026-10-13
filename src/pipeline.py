from __future__ import annotations

import argparse
from pathlib import Path

from src.audit_environment import audit
from src.data.build_manifest import build_manifest
from src.qwen_teacher.score_candidates import run_teacher_stage
from src.reporting import build_summary
from src.sam_masks.candidate_generator import run_candidate_stage
from src.sam_masks.mask_filter import run_filter_stage
from src.sam_masks.subject_mask import run_subject_stage
from src.token_labels.mask_to_tokens import run_token_label_stage
from src.visualization.render_sample import run_visualization_stage
from src.validate_outputs import validate_outputs


STAGES = ("audit", "manifest", "subject", "candidates", "filter", "teacher", "tokens", "visualize", "report", "validate")


def run_stage(stage: str, config_path: str | Path, overwrite: bool) -> None:
    if stage == "audit":
        _, ok = audit(config_path)
        if not ok:
            raise RuntimeError("Critical environment audit failed; inspect outputs/pilot_200/environment.json")
    elif stage == "manifest":
        build_manifest(config_path)
    elif stage == "subject":
        run_subject_stage(config_path, overwrite=overwrite)
    elif stage == "candidates":
        run_candidate_stage(config_path, overwrite=overwrite)
    elif stage == "filter":
        run_filter_stage(config_path, overwrite=overwrite)
    elif stage == "teacher":
        run_teacher_stage(config_path, overwrite=overwrite)
    elif stage == "tokens":
        run_token_label_stage(config_path, overwrite=overwrite)
    elif stage == "visualize":
        run_visualization_stage(config_path, overwrite=overwrite)
    elif stage == "report":
        build_summary(config_path)
    elif stage == "validate":
        report = validate_outputs(config_path)
        if not report["strict_ok"]:
            raise RuntimeError(f"Output validation failed: {report['errors']}")
    else:
        raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible 200-sample hierarchical VIS pilot")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else (args.stage,)
    for stage in selected:
        print(f"=== stage: {stage} ===", flush=True)
        run_stage(stage, args.config, args.overwrite)


if __name__ == "__main__":
    main()

