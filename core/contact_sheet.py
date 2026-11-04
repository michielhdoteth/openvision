"""Contact sheet generation: tile keyframes into grid images."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def create_contact_sheet(
    frames: list[dict],
    grid_size: tuple[int, int] = (3, 3),
    cell_width: int = 320,
    cell_height: int = 180,
    show_timestamps: bool = True,
    output_path: Optional[str] = None,
) -> str:
    """
    Tile keyframes into a contact sheet grid image.

    Args:
        frames: List of frame dicts with 'image' and 'timestamp' keys
        grid_size: (cols, rows) for the grid
        cell_width: Width of each cell in pixels
        cell_height: Height of each cell in pixels
        show_timestamps: Overlay timestamp text on each cell
        output_path: Where to save (generated if None)

    Returns:
        Path to saved contact sheet image
    """
    cols, rows = grid_size
    total_cells = cols * rows

    if len(frames) > total_cells:
        step = len(frames) / total_cells
        selected = [frames[int(i * step)] for i in range(total_cells)]
    else:
        selected = frames[:total_cells]
        # Pad with last frame if fewer than cells
        while len(selected) < total_cells:
            selected.append(selected[-1] if selected else frames[0])

    sheet_w = cols * cell_width
    sheet_h = rows * cell_height
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * cell_height

        img = frame.get("image")
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(cell_width / w, cell_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ox = x + (cell_width - new_w) // 2
        oy = y + (cell_height - new_h) // 2
        sheet[oy:oy + new_h, ox:ox + new_w] = resized

        if show_timestamps:
            ts = frame.get("timestamp_str", "")
            if not ts:
                ts = f"{frame.get('timestamp', 0):.1f}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(ts, font, font_scale, thickness)
            cv2.rectangle(sheet, (x + 4, y + cell_height - th - 8), (x + tw + 10, y + cell_height - 2), (0, 0, 0), -1)
            cv2.putText(sheet, ts, (x + 6, y + cell_height - 6), font, font_scale, (255, 255, 255), thickness)

        cv2.rectangle(sheet, (x, y), (x + cell_width - 1, y + cell_height - 1), (80, 80, 80), 1)

    if output_path is None:
        output_path = "contact_sheet.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def create_contact_sheet_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    grid_size: tuple[int, int] = (3, 3),
    num_frames: int = 9,
    max_resolution: int = 768,
) -> str:
    """
    Generate a contact sheet directly from a video file.

    Extracts evenly spaced frames and tiles them.

    Args:
        video_path: Path to video file
        output_path: Where to save (generated if None)
        grid_size: (cols, rows)
        num_frames: Number of frames to extract
        max_resolution: Max frame resolution

    Returns:
        Path to saved contact sheet image
    """
    from core.video import extract_frames

    frames = extract_frames(video_path, fps=0, max_frames=num_frames, max_resolution=max_resolution)
    if not frames:
        raise ValueError("No frames extracted from video")

    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_contact_sheet.jpg")

    return create_contact_sheet(frames, grid_size=grid_size, output_path=output_path)
