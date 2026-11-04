"""Smart frame sampling: scene detection, deduplication, keyframe selection."""
import cv2
import numpy as np
from typing import Optional
from core.video import extract_frames, probe, parse_time



def _hist_hsv(frame: np.ndarray) -> np.ndarray:
    """Compute normalized HSV histogram for a frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 30], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _motion_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute pixel-difference motion score between two frames (0=identical, 1=max change)."""
    if frame_a is None or frame_b is None:
        return 0.0
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(diff)) / 255.0


def sliding_window_dedup(
    frames: list[dict],
    window_size: int = 10,
    global_threshold: float = 0.92,
    local_threshold: float = 0.85,
    motion_threshold: float = 0.02,
) -> tuple[list[int], dict]:
    """
    Sliding-window deduplication with 3 channels:

    1. **Global channel**: Histogram correlation to globally unique reference.
       Rejects frames that match any previously kept frame too closely.
    2. **Settled-local channel**: Within the sliding window, tracks whether the
       scene has settled. Keeps a frame only if the window is "active" (content
       is still changing). Settled windows drop redundant frames.
    3. **Action channel**: Pixel-difference motion score. Keeps frames where
       motion exceeds a threshold, even if histogram similarity is high.

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        window_size: Number of frames in the sliding window
        global_threshold: 0-1, higher = more aggressive dedup (reject more)
        local_threshold: 0-1, threshold for local similarity within window
        motion_threshold: 0-1, minimum motion score to force-keep a frame

    Returns:
        Tuple of (kept_indices, stats_dict)
    """
    if len(frames) <= 1:
        return list(range(len(frames))), {"kept": len(frames), "dropped": 0}

    kept_indices = []
    kept_hists = []  # global reference histograms
    stats = {"kept": 0, "dropped_global": 0, "dropped_local": 0, "saved_motion": 0}

    for i, frame in enumerate(frames):
        img = frame.get("image")
        if img is None:
            continue

        hist = _hist_hsv(img)

        globally_similar = False
        for ref_hist in kept_hists:
            sim = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
            if sim >= global_threshold:
                globally_similar = True
                break

        if globally_similar:
            # Even if globally similar, keep if there's significant motion
            if kept_indices:
                prev_img = frames[kept_indices[-1]].get("image")
                motion = _motion_score(prev_img, img)
                if motion >= motion_threshold:
                    kept_indices.append(i)
                    kept_hists.append(hist)
                    stats["saved_motion"] += 1
                    stats["kept"] += 1
                    continue

            stats["dropped_global"] += 1
            continue

        # Check if the recent window is "settled" (all similar to each other)
        window_start = max(0, i - window_size)
        window_frames = frames[window_start:i]
        if len(window_frames) >= 3:
            window_hists = [_hist_hsv(f["image"]) for f in window_frames if f.get("image") is not None]
            if window_hists:
                local_sims = []
                for j in range(1, len(window_hists)):
                    s = cv2.compareHist(window_hists[j - 1], window_hists[j], cv2.HISTCMP_CORREL)
                    local_sims.append(s)
                avg_local = sum(local_sims) / len(local_sims) if local_sims else 1.0
                if avg_local >= local_threshold:
                    # Window is settled -- this frame is redundant locally
                    stats["dropped_local"] += 1
                    continue

        kept_indices.append(i)
        kept_hists.append(hist)
        stats["kept"] += 1

    return kept_indices, stats


def detect_scenes(
    frames: list[dict],
    threshold: float = 30.0,
    min_scene_gap: int = 2,
) -> list[int]:
    """
    Detect scene changes using HSV histogram comparison.

    Args:
        frames: List of frame dicts with 'image' key (np.ndarray)
        threshold: Histogram difference threshold (lower = more sensitive)
        min_scene_gap: Minimum frames between scene changes

    Returns:
        List of frame indices where scene changes occur
    """
    if len(frames) < 2:
        return []

    scene_indices = []
    prev_hist = None

    for i, frame in enumerate(frames):
        if frame["image"] is None:
            continue

        hsv = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2HSV)
        # Use all 3 HSV channels so uniform black/white frames are distinguishable
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 30], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

            if diff > threshold:
                if not scene_indices or (i - scene_indices[-1]) >= min_scene_gap:
                    scene_indices.append(i)

        prev_hist = hist

    return scene_indices


def deduplicate(
    frames: list[dict],
    indices: Optional[list[int]] = None,
    similarity_threshold: float = 0.95,
) -> list[int]:
    """
    Remove near-identical frames using histogram correlation.

    Args:
        frames: List of frame dicts with 'image' key
        indices: Subset of frame indices to filter (None = all)
        similarity_threshold: 0-1, higher = more aggressive dedup

    Returns:
        Filtered list of frame indices
    """
    if indices is None:
        indices = list(range(len(frames)))

    if len(indices) <= 1:
        return indices

    result = [indices[0]]
    ref_hist = None

    for idx in indices[1:]:
        frame = frames[idx]
        if frame["image"] is None:
            continue

        hsv = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2HSV)
        # Use all 3 HSV channels so uniform black/white frames are distinguishable
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 30], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        if ref_hist is not None:
            # Correlation: 1 = identical, 0 = unrelated
            similarity = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)

            if similarity < similarity_threshold:
                result.append(idx)
                ref_hist = hist
        else:
            ref_hist = hist

    return result


def smart_sample(
    path: str,
    fps: float = 0.5,
    max_frames: int = 60,
    max_resolution: int = 768,
    scene_threshold: float = 30.0,
    dedup_threshold: float = 0.95,
    vfr_mode: str = "unknown",
) -> list[dict]:
    """
    Full smart sampling pipeline:
    1. Extract frames at given FPS (up to max_frames)
    2. Detect scene changes
    3. Deduplicate near-identical frames
    4. Ensure scene change frames are included

    Returns:
        List of selected frame dicts with {index, timestamp, timestamp_str, image}
    """
    frames = extract_frames(path, fps=fps, max_frames=max_frames, max_resolution=max_resolution, vfr_mode=vfr_mode)

    if not frames:
        return []

    scene_indices = detect_scenes(frames, threshold=scene_threshold)

    uniform_indices = list(range(0, len(frames), max(1, len(frames) // min(max_frames, len(frames)))))
    uniform_indices = uniform_indices[:max_frames]

    all_indices = sorted(set(uniform_indices + scene_indices))

    # We dedup the uniform samples first, then merge scene changes
    deduped_uniform = deduplicate(frames, list(range(len(frames))), similarity_threshold=dedup_threshold)

    # Combine deduped uniform + scene changes, keep unique and in order
    final_indices = sorted(set(deduped_uniform + scene_indices))

    return [frames[i] for i in final_indices if i < len(frames)]


def find_keyframes_for_question(
    frames: list[dict],
    question_relevant_indices: Optional[list[int]] = None,
    max_keyframes: int = 10,
) -> list[dict]:
    """
    Select keyframes most relevant to a specific question.
    If no relevance hints, return evenly spaced keyframes.
    """
    if not frames:
        return []

    if question_relevant_indices:
        selected = [frames[i] for i in question_relevant_indices if i < len(frames)]
        return selected[:max_keyframes]

    step = max(1, len(frames) // min(max_keyframes, len(frames)))
    return [frames[i] for i in range(0, len(frames), step)][:max_keyframes]


def adaptive_sample(
    path: str,
    max_frames: int = 60,
    max_resolution: int = 768,
    dense_window: float = 3.0,  # seconds to sample densely around scene changes
    dense_fps: float = 2.0,     # FPS during dense windows
    base_fps: float = 0.25,     # FPS during static segments
    scene_threshold: float = 30.0,
    vfr_mode: str = "unknown",
) -> list[dict]:
    """
    Scene-adaptive frame sampling.

    Strategy:
    1. Do a coarse pass to detect all scene changes
    2. Sample densely (high FPS) around each scene change
    3. Sample sparsely (low FPS) during static segments
    4. Ensure total frames <= max_frames

    This gives more coverage near scene transitions where content
    actually changes, while skipping redundant frames during
    static talking-head or single-slide segments.
    """
    from core.video import extract_frames, probe, parse_time
    from core.sampling import detect_scenes

    try:
        meta = probe(path)
    except FileNotFoundError:
        return []

    duration = meta["duration_seconds"]
    video_fps = meta["fps"]

    # Use 1fps for scene detection (covers full video)
    coarse_frames = extract_frames(path, fps=1, max_frames=300, max_resolution=max_resolution, vfr_mode=vfr_mode)

    if not coarse_frames:
        return []

    scene_indices = detect_scenes(coarse_frames, threshold=scene_threshold)
    scene_times = [coarse_frames[i]["timestamp"] for i in scene_indices]

    # Each range is (start_time, end_time, fps)
    ranges = []

    if not scene_times:
        # No scene changes -- just sample evenly
        ranges.append((0.0, duration, base_fps))
    else:
        all_boundaries = [0.0] + scene_times + [duration]

        for i in range(len(all_boundaries) - 1):
            seg_start = all_boundaries[i]
            seg_end = all_boundaries[i + 1]
            seg_len = seg_end - seg_start

            is_dense = any(
                abs(st - (seg_start + seg_len / 2)) < dense_window / 2
                for st in scene_times
            )

            if is_dense or seg_len < dense_window * 2:
                # Dense sampling for the whole short segment
                ranges.append((seg_start, seg_end, max(base_fps, dense_fps)))
            else:
                ranges.append((seg_start, seg_end, base_fps))
                for st in scene_times:
                    if seg_start < st < seg_end:
                        dw_start = max(seg_start, st - dense_window / 2)
                        dw_end = min(seg_end, st + dense_window / 2)
                        ranges.append((dw_start, dw_end, dense_fps))

    ranges.sort(key=lambda r: r[0])
    merged = []
    for r in ranges:
        if merged and r[0] <= merged[-1][1]:
            existing = merged[-1]
            new_fps = max(existing[2], r[2])
            merged[-1] = (existing[0], max(existing[1], r[1]), new_fps)
        else:
            merged.append(r)

    # Reuse the single coarse decode. The prior implementation decoded the
    # complete video once for every scene range.
    #
    # Selection is an even lattice over the whole coarse pass, anchored at both
    # ends, plus a bonus trade of a few interior lattice points for the frames
    # nearest real scene cuts. Taking scene candidates alone (the old
    # `scene_candidates[:max_frames]`) front-loaded long videos: whenever the
    # opening was change-dense, every sample landed in the first few minutes and
    # the tail of the video went unsampled without any error.
    scene_candidates = [
        i for i, frame in enumerate(coarse_frames)
        if any(abs(frame["timestamp"] - scene_time) <= dense_window / 2 for scene_time in scene_times)
    ]
    budget = max(0, max_frames)
    lattice = _even_lattice(len(coarse_frames), budget)
    chosen = set(lattice)

    interior = list(range(1, max(1, budget - 1)))
    scene_slots = min(len(scene_candidates), len(interior) // 3)
    if scene_slots:
        scene_step = len(scene_candidates) / scene_slots
        for n in range(scene_slots):
            scene_index = scene_candidates[min(len(scene_candidates) - 1, int(n * scene_step))]
            if scene_index in chosen:
                continue
            chosen.discard(lattice[interior[int(n * len(interior) / scene_slots)]])
            chosen.add(scene_index)

    frames = [coarse_frames[i] for i in sorted(chosen)]

    frames.sort(key=lambda f: f["timestamp"])

    if len(frames) > max_frames:
        # Evenly subsample to fit max_frames
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]

    return frames


def _even_lattice(total: int, slots: int) -> list[int]:
    """Indices spread evenly over range(total), anchored at the first and last.

    Anchoring the ends is what guarantees a sample covers the whole source: a
    plain `int(i * total / slots)` lattice stops short of the final frame
    whenever total is not a multiple of slots, which is what let long videos
    look complete while missing their tail.
    """
    if slots <= 0 or total <= 0:
        return []
    if slots >= total:
        return list(range(total))
    if slots == 1:
        return [total // 2]
    last = total - 1
    return sorted({int(round(i * last / (slots - 1))) for i in range(slots)})
