"""Local HTML viewer for video observations and keyframes."""
from pathlib import Path
from typing import Optional
import json
import base64


def generate_viewer(
    frames: list[dict],
    transcript_segments: Optional[list] = None,
    summary: str = "",
    output_path: Optional[str] = None,
    title: str = "OpenVision Viewer",
) -> str:
    """
    Generate a self-contained HTML viewer with keyframes and transcript.

    Args:
        frames: List of frame dicts with 'image' (np.ndarray) and 'timestamp'
        transcript_segments: List of transcript segments with start, end, text
        summary: Observation summary text
        output_path: Where to save the HTML file
        title: Page title

    Returns:
        Path to generated HTML file
    """
    import cv2

    frame_cards = []
    for i, frame in enumerate(frames):
        img = frame.get("image")
        if img is None:
            continue
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("ascii")
        ts = frame.get("timestamp", 0)
        ts_str = frame.get("timestamp_str", f"{ts:.1f}s")
        frame_cards.append({
            "index": i,
            "timestamp": ts,
            "timestamp_str": ts_str,
            "image_b64": b64,
        })

    transcript_html = ""
    if transcript_segments:
        transcript_html = '<div class="transcript"><h2>Transcript</h2><div class="transcript-entries">'
        for seg in transcript_segments:
            start = getattr(seg, "start", 0)
            text = getattr(seg, "text", "")
            ts_m, ts_s = divmod(int(start), 60)
            ts_h, ts_m = divmod(ts_m, 60)
            ts_str = f"{ts_h:02d}:{ts_m:02d}:{ts_s:02d}" if ts_h else f"{ts_m:02d}:{ts_s:02d}"
            transcript_html += f'<div class="transcript-entry" data-time="{start:.1f}"><span class="ts">[{ts_str}]</span> {text}</div>'
        transcript_html += '</div></div>'

    cards_html = ""
    for fc in frame_cards:
        cards_html += f'''
        <div class="frame-card" data-time="{fc['timestamp']:.1f}">
            <img src="data:image/jpeg;base64,{fc['image_b64']}" alt="Frame {fc['index']}" loading="lazy">
            <div class="frame-ts">{fc['timestamp_str']}</div>
        </div>'''

    summary_html = f'<div class="summary"><h2>Summary</h2><p>{summary}</p></div>' if summary else ""

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
.summary {{ max-width: 800px; margin: 0 auto 30px; padding: 16px; background: #1a1a1a; border-radius: 8px; border-left: 3px solid #4fc3f7; }}
.summary h2 {{ font-size: 14px; color: #4fc3f7; margin-bottom: 8px; }}
.summary p {{ line-height: 1.6; }}
.frames {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; max-width: 1200px; margin: 0 auto 30px; }}
.frame-card {{ position: relative; border-radius: 8px; overflow: hidden; background: #1a1a1a; cursor: pointer; transition: transform 0.2s; }}
.frame-card:hover {{ transform: scale(1.02); }}
.frame-card img {{ width: 100%; display: block; }}
.frame-ts {{ position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,0.8); color: #4fc3f7; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: monospace; }}
.transcript {{ max-width: 800px; margin: 0 auto; }}
.transcript h2 {{ color: #4fc3f7; margin-bottom: 12px; }}
.transcript-entries {{ background: #1a1a1a; border-radius: 8px; padding: 16px; max-height: 400px; overflow-y: auto; }}
.transcript-entry {{ padding: 4px 0; line-height: 1.5; font-size: 14px; }}
.transcript-entry:hover {{ background: #222; }}
.ts {{ color: #4fc3f7; font-family: monospace; font-size: 12px; margin-right: 8px; }}
.frame-count {{ text-align: center; color: #666; font-size: 13px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="frame-count">{len(frame_cards)} keyframes</div>
{summary_html}
<div class="frames">{cards_html}</div>
{transcript_html}
</body>
</html>"""

    if output_path is None:
        output_path = "viewer.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
