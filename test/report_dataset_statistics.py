import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize WSI dataset statistics for the paper.")
    parser.add_argument("--dataset-root", default=str(WORKSPACE_ROOT / "data" / "private_dataset"))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "dataset_stats"))
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of cases to scan.")
    return parser


def try_import_openslide():
    try:
        import openslide  # type: ignore
        return openslide
    except Exception:
        return None


def inspect_slide(path: Path, openslide_mod) -> Dict[str, object]:
    stats: Dict[str, object] = {"path": str(path), "width": None, "height": None, "mpp_x": None, "mpp_y": None, "objective_power": None}
    if openslide_mod is None:
        return stats
    try:
        slide = openslide_mod.OpenSlide(str(path))
        stats["width"] = int(slide.dimensions[0])
        stats["height"] = int(slide.dimensions[1])
        props = slide.properties
        stats["mpp_x"] = float(props.get("openslide.mpp-x")) if props.get("openslide.mpp-x") else None
        stats["mpp_y"] = float(props.get("openslide.mpp-y")) if props.get("openslide.mpp-y") else None
        power = props.get("openslide.objective-power") or props.get("aperio.AppMag")
        stats["objective_power"] = float(power) if power else None
        slide.close()
    except Exception:
        return stats
    return stats


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openslide_mod = try_import_openslide()
    cases: List[Dict[str, object]] = []
    for case_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        case_id = case_dir.name
        unstained = case_dir / f"{case_id}_unstained.svs"
        he = case_dir / f"{case_id}_he.svs"
        if not unstained.exists():
            continue
        unstained_stats = inspect_slide(unstained, openslide_mod)
        he_stats = inspect_slide(he, openslide_mod) if he.exists() else {}
        cases.append(
            {
                "case_id": case_id,
                "unstained_path": str(unstained),
                "he_path": str(he) if he.exists() else "",
                "unstained_width": unstained_stats.get("width"),
                "unstained_height": unstained_stats.get("height"),
                "unstained_mpp_x": unstained_stats.get("mpp_x"),
                "unstained_mpp_y": unstained_stats.get("mpp_y"),
                "unstained_objective_power": unstained_stats.get("objective_power"),
                "he_width": he_stats.get("width"),
                "he_height": he_stats.get("height"),
                "he_mpp_x": he_stats.get("mpp_x"),
                "he_mpp_y": he_stats.get("mpp_y"),
                "he_objective_power": he_stats.get("objective_power"),
            }
        )
        if args.limit > 0 and len(cases) >= args.limit:
            break

    mpp_x = [float(item["unstained_mpp_x"]) for item in cases if isinstance(item.get("unstained_mpp_x"), (int, float))]
    mpp_y = [float(item["unstained_mpp_y"]) for item in cases if isinstance(item.get("unstained_mpp_y"), (int, float))]
    power = [float(item["unstained_objective_power"]) for item in cases if isinstance(item.get("unstained_objective_power"), (int, float))]
    widths = [int(item["unstained_width"]) for item in cases if isinstance(item.get("unstained_width"), int)]
    heights = [int(item["unstained_height"]) for item in cases if isinstance(item.get("unstained_height"), int)]

    summary = {
        "dataset_root": str(dataset_root),
        "num_cases": len(cases),
        "num_unstained_svs": len(cases),
        "num_he_svs": sum(1 for item in cases if item.get("he_path")),
        "mean_width": mean(widths) if widths else None,
        "mean_height": mean(heights) if heights else None,
        "mean_mpp_x": mean(mpp_x) if mpp_x else None,
        "mean_mpp_y": mean(mpp_y) if mpp_y else None,
        "mean_objective_power": mean(power) if power else None,
        "openslide_available": bool(openslide_mod is not None),
        "patch_cutting_method": "coarse overview screening + local refinement + paired patch export",
        "export_patch_size": 512,
        "patch_size": 256,
    }

    csv_path = output_dir / "dataset_statistics.csv"
    json_path = output_dir / "dataset_statistics.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "cases": cases}, file, ensure_ascii=False, indent=2)

    print(f"dataset_statistics_csv={csv_path}")
    print(f"dataset_statistics_json={json_path}")
    print("summary=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
