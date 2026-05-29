import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run top-k sensitivity sweep for focused virtual staining.")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT.parent / "data" / "private_dataset"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT.parent / "outputs" / "topk_sweep"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=20, help="Number of cases per setting.")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5], help="Top-k values to sweep.")
    parser.add_argument("--threshold", type=float, default=0.34)
    parser.add_argument("--grid", type=int, default=5)
    return parser


def run_one(dataset_root: str, output_dir: Path, model_path: str | None, top_k: int, limit: int, threshold: float, grid: int) -> None:
    cmd = [
        sys.executable,
        str(CURRENT_DIR / "run_pathology_batch_eval.py"),
        "--dataset-root",
        dataset_root,
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(top_k),
        "--threshold",
        str(threshold),
        "--grid",
        str(grid),
        "--limit",
        str(limit),
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])
    subprocess.run(cmd, check=True)


def extract_summary(run_dir: Path) -> Dict[str, float]:
    summary_path = run_dir / "batch_metrics.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    summary = dict(payload.get("summary", {}))
    return {
        "selected_highres_ratio_mean": float(summary.get("selected_highres_ratio_mean", 0.0)),
        "hard_mask_coverage_mean": float(summary.get("hard_mask_coverage_mean", 0.0)),
        "soft_attention_mean_mean": float(summary.get("soft_attention_mean_mean", 0.0)),
        "selected_score_mean_mean": float(summary.get("selected_score_mean_mean", 0.0)),
        "selected_confidence_mean_mean": float(summary.get("selected_confidence_mean_mean", 0.0)),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for top_k in args.top_ks:
        run_dir = output_root / f"topk_{top_k}"
        print(f"top_k={top_k} run_dir={run_dir}")
        run_status = "ok"
        try:
            run_one(args.dataset_root, run_dir, args.model_path, top_k, args.limit, args.threshold, args.grid)
        except subprocess.CalledProcessError as exc:
            run_status = f"failed:{exc.returncode}"
            print(f"warning=topk_run_failed top_k={top_k} returncode={exc.returncode}")
        metrics = extract_summary(run_dir)
        rows.append({"top_k": top_k, "run_status": run_status, **metrics})

    csv_path = output_root / "topk_sweep_summary.csv"
    json_path = output_root / "topk_sweep_summary.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "top_k",
                "run_status",
                "selected_highres_ratio_mean",
                "hard_mask_coverage_mean",
                "soft_attention_mean_mean",
                "selected_score_mean_mean",
                "selected_confidence_mean_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"runs": rows}, file, ensure_ascii=False, indent=2)

    print(f"topk_sweep_summary_csv={csv_path}")
    print(f"topk_sweep_summary_json={json_path}")


if __name__ == "__main__":
    main()
