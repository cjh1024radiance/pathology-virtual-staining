import html
import json
import os
from pathlib import Path
from typing import Dict, List


def write_pathology_report(result_json_path: str) -> str:
    result_path = Path(result_json_path)
    output_dir = result_path.parent
    with open(result_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    def rel(path: str | None) -> str:
        if not path:
            return ""
        try:
            return os.path.relpath(path, output_dir).replace("\\", "/")
        except Exception:
            return path

    rois: List[Dict[str, object]] = list(data.get("rois", []))
    artifacts: List[Dict[str, object]] = list(data.get("artifacts", []))
    metrics: Dict[str, object] = dict(data.get("metrics", {}))
    api_usage: Dict[str, object] = dict(data.get("api_usage", {}))
    llm_overview: Dict[str, object] = dict(data.get("llm_overview", {}))
    suspicious_regions = llm_overview.get("suspicious_regions", [])

    roi_cards = []
    for roi in sorted(rois, key=lambda item: float(item.get("final_score", 0.0)), reverse=True):
        tile_id = roi.get("tile_id", "?")
        roi_dir = output_dir / f"roi_{int(tile_id):02d}" if isinstance(tile_id, int) else output_dir
        patch_path = rel(str(roi_dir / "patch.png"))
        stain_path = rel(str(roi_dir / "stain.png"))
        weight_path = rel(str(roi_dir / "routing_weight.png"))
        trace = "<br>".join(html.escape(str(item)) for item in roi.get("stage_trace", []))
        roi_cards.append(
            f"""
            <div class="roi-card">
              <div class="roi-head">
                <h3>ROI {tile_id}</h3>
                <span class="badge">{html.escape(str(roi.get('suggested_magnification', '10x')))}</span>
              </div>
              <p><strong>Score:</strong> {float(roi.get('final_score', 0.0)):.3f}</p>
              <p><strong>Confidence:</strong> {float(roi.get('lesion_confidence', 0.0)):.3f}</p>
              <p><strong>Type:</strong> {html.escape(str(roi.get('suspicion_type', 'unspecified')))}</p>
              <p><strong>Explanation:</strong> {html.escape(str(roi.get('explanation', '')))}</p>
              <p><strong>Visited round:</strong> {html.escape(str(roi.get('visited_round', 0)))}</p>
              <div class="roi-grid">
                <figure><img src="{patch_path}" alt="ROI patch"><figcaption>Original ROI</figcaption></figure>
                <figure><img src="{stain_path}" alt="ROI stain"><figcaption>Stained ROI</figcaption></figure>
                <figure><img src="{weight_path}" alt="ROI weight"><figcaption>Routing weight</figcaption></figure>
              </div>
              <details><summary>Stage trace</summary><div class="trace">{trace}</div></details>
            </div>
            """
        )

    report_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Pathology Navigation Report</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; background: #f5f6f8; color: #1f2937; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .hero {{ background: white; border-radius: 16px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }}
    .metric {{ background: #eef2ff; border-radius: 12px; padding: 14px; }}
    .gallery {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; margin-top: 20px; }}
    figure {{ margin: 0; background: white; border-radius: 14px; padding: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }}
    figure img {{ width: 100%; border-radius: 10px; background: #e5e7eb; }}
    figcaption {{ margin-top: 8px; font-size: 14px; color: #4b5563; }}
    .roi-section {{ margin-top: 28px; }}
    .roi-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .roi-card {{ background: white; border-radius: 16px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }}
    .roi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .roi-head {{ display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: #111827; color: white; border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
    .trace {{ margin-top: 10px; font-size: 13px; line-height: 1.5; color: #374151; }}
    .code {{ font-family: Consolas, monospace; font-size: 13px; white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>病理导航与虚拟染色报告</h1>
      <p><strong>Input:</strong> {html.escape(str(data.get("input_path", "")))}</p>
      <p><strong>Output dir:</strong> {html.escape(str(output_dir))}</p>
      <p><strong>Overall assessment:</strong> {html.escape(str(llm_overview.get("overall_assessment", "N/A")))}</p>
      <p><strong>Focus recommendation:</strong> {html.escape(str(llm_overview.get("focus_recommendation", "N/A")))}</p>
      <p><strong>Summary:</strong> {html.escape(str(llm_overview.get("summary", "N/A")))}</p>
      <div class="metrics">
        <div class="metric"><strong>API total calls</strong><br>{api_usage.get("total_calls", 0)}</div>
        <div class="metric"><strong>Selected high-res ROI</strong><br>{metrics.get("selected_highres_rois", 0)}</div>
        <div class="metric"><strong>Visited ROI</strong><br>{metrics.get("visited_rois", 0)}</div>
        <div class="metric"><strong>Compute reduction</strong><br>{float(metrics.get("estimated_compute_reduction_percent", 0.0)):.2f}%</div>
      </div>
    </section>

    <section class="roi-section">
      <h2>LLM Suspicious Region Summary</h2>
      <div class="roi-list">
        {''.join(
            f"<div class='roi-card'><p><strong>Position:</strong> {html.escape(str(item.get('position', 'N/A')))}</p>"
            f"<p><strong>Suspicion level:</strong> {html.escape(str(item.get('suspicion_level', 'N/A')))}</p>"
            f"<p><strong>Reason:</strong> {html.escape(str(item.get('reason', '')))}</p></div>"
            for item in suspicious_regions if isinstance(item, dict)
        )}
      </div>
    </section>

    <section class="gallery">
      <figure><img src="{rel(data.get('overview_risk_heatmap_path'))}" alt="Overview risk heatmap"><figcaption>Overview risk heatmap</figcaption></figure>
      <figure><img src="{rel(data.get('roi_overlay_path'))}" alt="ROI overlay"><figcaption>ROI selection and navigation rounds</figcaption></figure>
      <figure><img src="{rel(data.get('routing_hard_mask_path'))}" alt="Hard routing mask"><figcaption>Hard routing mask</figcaption></figure>
      <figure><img src="{rel(data.get('routing_soft_attention_path'))}" alt="Soft attention map"><figcaption>Soft attention map</figcaption></figure>
      <figure><img src="{rel(data.get('lowres_stain_path'))}" alt="Lowres stain"><figcaption>Whole-slide low-resolution stain preview</figcaption></figure>
      <figure><img src="{rel(data.get('fused_stain_path'))}" alt="Fused stain"><figcaption>Final fused stain result</figcaption></figure>
      <figure><img src="{rel(data.get('fused_roi_overlay_path'))}" alt="Fused ROI overlay"><figcaption>Lesion ROI overlay on stained result</figcaption></figure>
    </section>

    <section class="roi-section">
      <h2>ROI Detail</h2>
      <div class="roi-list">
        {''.join(roi_cards)}
      </div>
    </section>

    <section class="roi-section">
      <h2>API Trace</h2>
      <div class="code">{html.escape(json.dumps(api_usage, ensure_ascii=False, indent=2))}</div>
    </section>
  </div>
</body>
</html>
"""
    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_html)
    return str(report_path)


__all__ = ["write_pathology_report"]
