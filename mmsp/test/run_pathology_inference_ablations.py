import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

INFERENCE_ABLATION_ORDER = [
    "full",
    "wo_morphology_aware_prompt",
    "wo_local_roi_api_review",
    "wo_local_refinement",
    "wo_routing_constraint",
]

INFERENCE_ABLATION_DESCRIPTIONS = {
    "full": "Full",
    "wo_morphology_aware_prompt": "w/o morphology-aware prompt",
    "wo_local_roi_api_review": "w/o local ROI API review",
    "wo_local_refinement": "w/o local refinement",
    "wo_routing_constraint": "w/o routing constraint",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference-side ablations for ROI selection and focused staining.")
    parser.add_argument("--manifest", default=None, help="Manifest with cases and optional ROI annotations.")
    parser.add_argument("--dataset-root", default=None, help="Dataset root to auto-discover unstained cases.")
    parser.add_argument(
        "--annotation-manifest",
        default=None,
        help="Optional annotation manifest used to merge roi_boxes / lesion_mask_path into auto-discovered cases.",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT.parent / "outputs" / "test_ablation"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.34)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20, help="Number of cases per ablation run. Default: 20.")
    return parser


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT.parent / path).resolve()


def load_manifest_cases(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, dict):
        cases = raw.get("cases", [])
    elif isinstance(raw, list):
        cases = raw
    else:
        raise ValueError("Manifest must be a list or object with 'cases'.")
    return [dict(item) for item in cases if isinstance(item, dict)]


def auto_discover_cases(dataset_root: str, limit: int) -> List[Dict[str, object]]:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise ValueError(f"dataset_root does not exist: {dataset_root}")
    cases: List[Dict[str, object]] = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        case_id = case_dir.name
        unstained = case_dir / f"{case_id}_unstained.svs"
        if unstained.exists():
            cases.append(
                {
                    "case_id": case_id,
                    "input_path": str(unstained),
                    "target_path": "",
                    "roi_boxes": [],
                    "lesion_mask_path": "",
                }
            )
        if limit > 0 and len(cases) >= limit:
            break
    if not cases:
        raise ValueError(f"No *_unstained.svs cases found under {root}")
    return cases


def build_run_manifest(args: argparse.Namespace) -> Path:
    if args.manifest:
        manifest_path = resolve_path(args.manifest)
        cases = load_manifest_cases(str(manifest_path))
        if args.limit is not None and args.limit > 0:
            cases = cases[: args.limit]
    elif args.dataset_root:
        cases = auto_discover_cases(args.dataset_root, args.limit or 20)
    else:
        raise ValueError("Please provide either --manifest or --dataset-root.")

    if args.annotation_manifest:
        annotation_cases = load_manifest_cases(str(resolve_path(args.annotation_manifest)))
        annotation_map = {str(item.get('case_id', '')).strip(): item for item in annotation_cases}
        merged_cases: List[Dict[str, object]] = []
        for case in cases:
            case_id = str(case.get("case_id", "")).strip()
            merged = dict(case)
            ann = annotation_map.get(case_id)
            if ann:
                if isinstance(ann.get("roi_boxes"), list):
                    merged["roi_boxes"] = ann.get("roi_boxes", [])
                if isinstance(ann.get("lesion_mask_path"), str):
                    merged["lesion_mask_path"] = ann.get("lesion_mask_path", "")
                if isinstance(ann.get("target_path"), str) and ann.get("target_path", "").strip():
                    merged["target_path"] = ann["target_path"]
            merged_cases.append(merged)
        cases = merged_cases

    temp_dir = Path(tempfile.mkdtemp(prefix="pathology_inference_ablation_", dir=str(PROJECT_ROOT.parent)))
    manifest_path = temp_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump({"cases": cases}, file, ensure_ascii=False, indent=2)
    return manifest_path


def run_one(args: argparse.Namespace, mode: str, run_dir: Path, manifest_path: Path) -> None:
    cmd = [
        sys.executable,
        str(CURRENT_DIR / "run_pathology_batch_eval.py"),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(run_dir),
        "--ablation-mode",
        mode,
        "--top-k",
        str(args.top_k),
        "--iou-threshold",
        str(args.iou_threshold),
        "--threshold",
        str(args.threshold),
        "--grid",
        str(args.grid),
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.limit is not None and args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    subprocess.run(cmd, check=True)


def extract_summary(run_dir: Path) -> Dict[str, float]:
    summary_path = run_dir / "batch_metrics.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    summary = dict(payload.get("summary", {}))
    metrics: Dict[str, float] = {
        "selected_highres_ratio_mean": float(summary.get("selected_highres_ratio_mean", 0.0)),
        "selected_highres_ratio_std": float(summary.get("selected_highres_ratio_std", 0.0)),
        "hard_mask_coverage_mean": float(summary.get("hard_mask_coverage_mean", 0.0)),
        "hard_mask_coverage_std": float(summary.get("hard_mask_coverage_std", 0.0)),
        "soft_attention_mean_mean": float(summary.get("soft_attention_mean_mean", 0.0)),
        "soft_attention_mean_std": float(summary.get("soft_attention_mean_std", 0.0)),
        "selected_score_mean_mean": float(summary.get("selected_score_mean_mean", 0.0)),
        "selected_score_mean_std": float(summary.get("selected_score_mean_std", 0.0)),
        "selected_confidence_mean_mean": float(summary.get("selected_confidence_mean_mean", 0.0)),
        "selected_confidence_mean_std": float(summary.get("selected_confidence_mean_std", 0.0)),
        "case_success_rate": float(summary.get("case_success_mean", 0.0)),
        "valid_roi_ratio": float(summary.get("valid_roi_ratio_mean", 0.0)),
    }
    if float(summary.get("roi_annotation_cases", 0.0)) > 0.0:
        metrics.update(
            {
                "roi_recall_at_3": float(summary.get("roi_recall_at_k_mean", 0.0)),
                "roi_precision_at_3": float(summary.get("roi_precision_at_k_mean", 0.0)),
                "roi_mean_best_iou": float(summary.get("roi_mean_best_iou_mean", 0.0)),
            }
        )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    run_manifest_path = build_run_manifest(args)

    for index, mode in enumerate(INFERENCE_ABLATION_ORDER, start=1):
        run_dir = output_root / mode
        print(f"[{index}/{len(INFERENCE_ABLATION_ORDER)}] inference={mode} ({INFERENCE_ABLATION_DESCRIPTIONS[mode]})")
        run_status = "ok"
        try:
            run_one(args, mode, run_dir, run_manifest_path)
        except subprocess.CalledProcessError as exc:
            run_status = f"failed:{exc.returncode}"
            print(f"warning=inference_ablation_failed mode={mode} returncode={exc.returncode}")
        metrics = extract_summary(run_dir)
        rows.append(
            {
                "setting": mode,
                "description": INFERENCE_ABLATION_DESCRIPTIONS[mode],
                "run_status": run_status,
                **metrics,
            }
        )

    csv_path = output_root / "inference_ablation_summary.csv"
    json_path = output_root / "inference_ablation_summary.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "setting",
                "description",
                "run_status",
                "selected_highres_ratio_mean",
                "selected_highres_ratio_std",
                "hard_mask_coverage_mean",
                "hard_mask_coverage_std",
                "soft_attention_mean_mean",
                "soft_attention_mean_std",
                "selected_score_mean_mean",
                "selected_score_mean_std",
                "selected_confidence_mean_mean",
                "selected_confidence_mean_std",
                "case_success_rate",
                "valid_roi_ratio",
                "roi_recall_at_3",
                "roi_precision_at_3",
                "roi_mean_best_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"runs": rows}, file, ensure_ascii=False, indent=2)

    print(f"inference_ablation_summary_csv={csv_path}")
    print(f"inference_ablation_summary_json={json_path}")


if __name__ == "__main__":
    main()
