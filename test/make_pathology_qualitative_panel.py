import argparse
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose a qualitative montage from batch inference outputs.")
    parser.add_argument("--input-root", required=True, help="Batch output root containing per-case subfolders.")
    parser.add_argument("--output-path", required=True, help="Output PNG path for the composed panel.")
    parser.add_argument("--case-ids", nargs="+", default=None, help="Optional explicit case ids to include.")
    parser.add_argument("--max-cases", type=int, default=4, help="Maximum number of cases to include.")
    return parser


def load_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize(size, Image.Resampling.LANCZOS)


def find_case_dirs(root: Path, case_ids: List[str] | None, max_cases: int) -> List[Path]:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if case_ids:
        wanted = {case_id.strip() for case_id in case_ids}
        candidates = [p for p in candidates if p.name in wanted]
    candidates = sorted(candidates, key=lambda p: p.name)
    return candidates[:max_cases]


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(input_root, args.case_ids, args.max_cases)
    if not case_dirs:
        raise ValueError("No case directories found for montage.")

    cols = 4
    thumb_w, thumb_h = 360, 240
    header_h = 44
    pad = 10
    rows = len(case_dirs)
    panel = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + header_h) + (rows + 1) * pad), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    titles = [
        "overview_risk_heatmap.png",
        "roi_overlay.png",
        "routing_hard_mask.png",
        "fused_roi_overlay.png",
    ]
    labels = ["Overview", "ROI Overlay", "Hard Mask", "Fused Overlay"]

    for row_idx, case_dir in enumerate(case_dirs):
        y0 = pad + row_idx * (thumb_h + header_h + pad)
        draw.text((pad, y0), f"Case: {case_dir.name}", fill=(0, 0, 0))
        for col_idx, (fname, label) in enumerate(zip(titles, labels)):
            x0 = pad + col_idx * (thumb_w + pad)
            y_img = y0 + header_h
            image_path = case_dir / fname
            if image_path.exists():
                image = load_image(image_path, (thumb_w, thumb_h))
            else:
                image = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                tmp = ImageDraw.Draw(image)
                tmp.text((20, thumb_h // 2 - 10), "missing", fill=(150, 150, 150))
            panel.paste(image, (x0, y_img))
            draw.rectangle([x0, y_img, x0 + thumb_w, y_img + thumb_h], outline=(40, 40, 40), width=1)
            draw.text((x0 + 8, y_img + 6), label, fill=(255, 255, 255))

    panel.save(output_path)
    print(f"qualitative_panel={output_path}")


if __name__ == "__main__":
    main()
