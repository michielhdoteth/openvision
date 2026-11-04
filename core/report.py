"""Report generation: keep/drop visualization for frame selection decisions."""
import cv2
import base64
from pathlib import Path
from typing import Optional


def generate_keep_drop_report(
    all_frames: list[dict],
    kept_indices: list[int],
    dedup_stats: Optional[dict] = None,
    output_path: Optional[str] = None,
    title: str = "Frame Selection Report",
) -> str:
    """
    Generate an HTML report visualizing which frames were kept vs dropped.

    Shows a timeline with green (kept) and red (dropped) markers, plus
    thumbnails of kept frames and a stats summary.

    Args:
        all_frames: All extracted frames with 'image', 'timestamp'
        kept_indices: Indices of frames that were kept
        dedup_stats: Optional dict with dedup statistics
        output_path: Where to save the HTML
        title: Report title

    Returns:
        Path to generated report
    """
    kept_set = set(kept_indices)
    total = len(all_frames)
    kept_count = len(kept_indices)
    dropped_count = total - kept_count

    timeline_data = []
    for i, frame in enumerate(all_frames):
        ts = frame.get("timestamp", 0)
        kept = i in kept_set
        timeline_data.append({"index": i, "time": ts, "kept": kept})

    kept_thumbs = ""
    for idx in kept_indices:
        if idx >= len(all_frames):
            continue
        frame = all_frames[idx]
        img = frame.get("image")
        if img is None:
            continue
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode("ascii")
        ts = frame.get("timestamp", 0)
        ts_str = frame.get("timestamp_str", f"{ts:.1f}s")
        kept_thumbs += f'<div class="thumb"><img src="data:image/jpeg;base64,{b64}" alt="Frame {idx}"><span>{ts_str}</span></div>'

    stats_html = ""
    if dedup_stats:
        stats_html = "<ul>"
        for k, v in dedup_stats.items():
            stats_html += f"<li><strong>{k}:</strong> {v}</li>"
        stats_html += "</ul>"

    timeline_json = str(timeline_data).replace("'", '"')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 20px; color: #4fc3f7; }}
.stats {{ max-width: 600px; margin: 0 auto 30px; padding: 16px; background: #1a1a1a; border-radius: 8px; display: flex; justify-content: space-around; text-align: center; }}
.stat {{ }}
.stat-num {{ font-size: 28px; font-weight: bold; }}
.stat-label {{ font-size: 12px; color: #888; margin-top: 4px; }}
.kept-color {{ color: #4caf50; }}
.dropped-color {{ color: #f44336; }}
.timeline {{ max-width: 1000px; margin: 0 auto 30px; padding: 16px; background: #1a1a1a; border-radius: 8px; position: relative; height: 60px; }}
.timeline-bar {{ display: flex; align-items: center; height: 100%; gap: 1px; padding: 0 10px; }}
.bar {{ flex: 1; min-width: 2px; height: 20px; border-radius: 1px; }}
.bar.kept {{ background: #4caf50; }}
.bar.dropped {{ background: #f44336; opacity: 0.4; }}
.timeline-label {{ position: absolute; bottom: 4px; left: 10px; right: 10px; display: flex; justify-content: space-between; font-size: 10px; color: #666; }}
.thumbs-title {{ max-width: 1000px; margin: 0 auto 12px; color: #4fc3f7; font-size: 14px; }}
.thumbs {{ display: flex; flex-wrap: wrap; gap: 8px; max-width: 1000px; margin: 0 auto; }}
.thumb {{ position: relative; width: 150px; }}
.thumb img {{ width: 100%; border-radius: 4px; }}
.thumb span {{ display: block; text-align: center; font-size: 11px; color: #888; margin-top: 2px; font-family: monospace; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stats">
    <div class="stat"><div class="stat-num">{total}</div><div class="stat-label">Total Frames</div></div>
    <div class="stat"><div class="stat-num kept-color">{kept_count}</div><div class="stat-label">Kept</div></div>
    <div class="stat"><div class="stat-num dropped-color">{dropped_count}</div><div class="stat-label">Dropped</div></div>
    <div class="stat"><div class="stat-num">{kept_count/total*100:.0f}%</div><div class="stat-label">Retention</div></div>
</div>
<div class="timeline">
    <div class="timeline-bar" id="timeline-bar"></div>
    <div class="timeline-label"><span>0:00</span><span>{all_frames[-1].get('timestamp', 0):.0f}s</span></div>
</div>
{f'<div class="thumbs-title">Kept Frames ({kept_count})</div>' if kept_thumbs else ''}
<div class="thumbs">{kept_thumbs}</div>
{f'<div style="max-width:1000px;margin:20px auto;padding:16px;background:#1a1a1a;border-radius:8px;"><h2 style="color:#4fc3f7;font-size:14px;margin-bottom:8px;">Dedup Stats</h2>{stats_html}</div>' if stats_html else ''}
<script>
const data = {timeline_json};
const bar = document.getElementById('timeline-bar');
data.forEach(d => {{
    const div = document.createElement('div');
    div.className = 'bar ' + (d.kept ? 'kept' : 'dropped');
    div.title = d.time.toFixed(1) + 's - ' + (d.kept ? 'KEPT' : 'DROPPED');
    bar.appendChild(div);
}});
</script>
</body>
</html>"""

    if output_path is None:
        output_path = "report.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
