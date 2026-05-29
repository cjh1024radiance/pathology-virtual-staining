import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
from train.run_virtual_stain_training import ABLATION_MODE_DESCRIPTIONS


ABLATION_ORDER = [
    "full",
    "wo_attention_conditioned_input",
    "wo_focus_reconstruction",
    "wo_clinical_consistency",
    "wo_he_aux_head",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run five training ablations for lesion-aware virtual staining.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "training_config.json"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT.parent / "outputs" / "train_ablation"))
    return parser


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run_one(config_path: Path) -> None:
    cmd = [sys.executable, str(CURRENT_DIR / "run_virtual_stain_training.py"), "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def extract_metrics(run_dir: Path) -> Dict[str, float]:
    test_metrics_path = run_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        return {}
    with open(test_metrics_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    metrics = dict(payload.get("test_metrics", {}))
    return {
        "weighted_l1": float(metrics.get("weighted_l1", 0.0)),
        "lesion_region_ssim": float(metrics.get("lesion_region_ssim", 0.0)),
        "clinical_consistency_loss": float(metrics.get("clinical_consistency_loss", 0.0)),
        "he_aux_loss": float(metrics.get("he_aux_loss", 0.0)),
    }


def main() -> None:
    args = build_parser().parse_args()
    base_cfg = load_json(args.config)
    output_root = Path(args.output_root)
    config_dir = output_root / "configs"
    summary_rows: List[Dict[str, object]] = []

    prepared_dir = Path(str(base_cfg.get("prepare", {}).get("prepared_dir", "")))
    prepared_exists = prepared_dir.exists() if prepared_dir else False

    for index, run_name in enumerate(ABLATION_ORDER, start=1):
        cfg = deepcopy(base_cfg)
        cfg.setdefault("runtime", {})
        cfg["runtime"]["ablation_mode"] = run_name
        run_dir = output_root / run_name
        cfg["runtime"]["output_dir"] = str(run_dir)

        cfg.setdefault("prepare", {})
        if index > 1 or prepared_exists:
            cfg["prepare"]["overwrite_prepared"] = False

        config_path = config_dir / f"{run_name}.json"
        save_json(config_path, cfg)
        print(f"[{index}/{len(ABLATION_ORDER)}] training={run_name} ({ABLATION_MODE_DESCRIPTIONS[run_name]})")
        run_one(config_path)

        metrics = extract_metrics(run_dir)
        summary_rows.append(
            {
                "setting": run_name,
                "description": ABLATION_MODE_DESCRIPTIONS[run_name],
                **metrics,
            }
        )

    summary_csv = output_root / "ablation_summary.csv"
    summary_json = output_root / "ablation_summary.json"

    with open(summary_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "setting",
                "description",
                "weighted_l1",
                "lesion_region_ssim",
                "clinical_consistency_loss",
                "he_aux_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump({"runs": summary_rows}, file, ensure_ascii=False, indent=2)

    print(f"ablation_summary_csv={summary_csv}")
    print(f"ablation_summary_json={summary_json}")


if __name__ == "__main__":
    main()
