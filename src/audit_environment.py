from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils import atomic_write_json, load_config, sha256_file


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15)
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            rows.append(dict(zip(("index", "name", "memory_total_mib", "memory_free_mib", "driver_version"), fields)))
    return rows


def _check_qwen_checkpoint(path: Path) -> dict[str, Any]:
    index_path = path / "model.safetensors.index.json"
    missing: list[str] = []
    shards: list[str] = []
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0]
    else:
        missing.append("model.safetensors.index.json")
    for required in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if not (path / required).is_file():
            missing.append(required)
    return {"path": str(path), "exists": path.is_dir(), "shards": shards, "missing": sorted(set(missing))}


def audit(config_path: str | Path) -> tuple[dict[str, Any], bool]:
    config = load_config(config_path)
    paths = config["paths"]
    manifest = Path(paths["training_manifest"])
    image_root = Path(paths["image_root"])
    sam_checkpoint = Path(paths["sam_checkpoint"])
    qwen_checkpoint = Path(paths["qwen_checkpoint"])
    qwen = _check_qwen_checkpoint(qwen_checkpoint)
    report: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "numpy", "Pillow", "PyYAML", "pycocotools", "opencv-python", "segment-anything", "qwen-vl-utils")
        },
        "modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "cv2", "pycocotools", "segment_anything", "qwen_vl_utils")
        },
        "gpu": _gpu_inventory(),
        "training_manifest": {
            "path": str(manifest),
            "exists": manifest.is_file(),
            "sha256": sha256_file(manifest) if manifest.is_file() else None,
        },
        "image_root": {"path": str(image_root), "exists": image_root.is_dir()},
        "sam_checkpoint": {
            "path": str(sam_checkpoint),
            "exists": sam_checkpoint.is_file(),
            "size": sam_checkpoint.stat().st_size if sam_checkpoint.is_file() else None,
        },
        "qwen_checkpoint": qwen,
        "qwen_preprocessing": {
            "min_pixels": config["qwen"]["min_pixels"],
            "max_pixels": config["qwen"]["max_pixels"],
            "patch_size": config["qwen"]["patch_size"],
            "spatial_merge_size": config["qwen"]["spatial_merge_size"],
            "source": "GAZE qwen-vl-finetune/scripts/sft_7b.sh plus checkpoint config",
        },
    }
    critical_ok = (
        report["training_manifest"]["exists"]
        and report["image_root"]["exists"]
        and report["sam_checkpoint"]["exists"]
        and qwen["exists"]
        and not qwen["missing"]
        and report["modules"]["segment_anything"]
        and report["modules"]["torch"]
        and report["modules"]["transformers"]
    )
    report["critical_ok"] = bool(critical_ok)
    output_dir = Path(paths["output_dir"])
    atomic_write_json(output_dir / "environment.json", report)
    return report, bool(critical_ok)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all external inputs before running the pilot")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    args = parser.parse_args()
    report, ok = audit(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

