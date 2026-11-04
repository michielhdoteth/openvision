"""WebVTT/SRT caption parsing and deduplication for YouTube auto-subs."""
import re
from dataclasses import dataclass


@dataclass
class CaptionSegment:
    """A single caption segment with timing."""
    start: float  # seconds
    end: float    # seconds
    text: str


def parse_srt(content: str) -> list[CaptionSegment]:
    """Parse SRT content into caption segments."""
    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())
    timestamp_pattern = re.compile(
        r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})'
    )
    for block in blocks:
        lines = block.strip().split('\n')
        for i, line in enumerate(lines):
            m = timestamp_pattern.search(line)
            if m:
                start = _ts_to_seconds(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                end = _ts_to_seconds(int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8)))
                text_lines = []
                for tl in lines[i + 1:]:
                    tl = tl.strip()
                    if not tl:
                        break
                    tl = re.sub(r'<[^>]+>', '', tl)
                    text_lines.append(tl)
                text = ' '.join(text_lines).strip()
                if text:
                    segments.append(CaptionSegment(start=start, end=end, text=text))
    return segments


def parse_vtt(content: str) -> list[CaptionSegment]:
    """Parse WebVTT content into caption segments.

    Handles both standard VTT and YouTube auto-generated captions.
    YouTube auto-subs have rolling duplicate text that needs dedup.
    """
    segments = []

    # Split into blocks (separated by blank lines)
    blocks = re.split(r'\n\s*\n', content.strip())

    timestamp_pattern = re.compile(
        r'(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})'
    )
    # Also match MM:SS format
    timestamp_pattern_short = re.compile(
        r'(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2})[.,](\d{3})'
    )

    for block in blocks:
        lines = block.strip().split('\n')

        for i, line in enumerate(lines):
            m = timestamp_pattern.search(line)
            if m:
                start = _ts_to_seconds(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                end = _ts_to_seconds(int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8)))
                # Text is everything after this line until blank line
                text_lines = []
                for tl in lines[i + 1:]:
                    tl = tl.strip()
                    if not tl or tl.startswith('NOTE'):
                        break
                    # Remove VTT tags like <c> </c>
                    tl = re.sub(r'<[^>]+>', '', tl)
                    text_lines.append(tl)
                text = ' '.join(text_lines).strip()
                if text:
                    segments.append(CaptionSegment(start=start, end=end, text=text))
                continue

            m = timestamp_pattern_short.search(line)
            if m:
                start = _ts_to_seconds(0, int(m.group(1)), int(m.group(2)), int(m.group(3)))
                end = _ts_to_seconds(0, int(m.group(4)), int(m.group(5)), int(m.group(6)))
                text_lines = []
                for tl in lines[i + 1:]:
                    tl = tl.strip()
                    if not tl or tl.startswith('NOTE'):
                        break
                    tl = re.sub(r'<[^>]+>', '', tl)
                    text_lines.append(tl)
                text = ' '.join(text_lines).strip()
                if text:
                    segments.append(CaptionSegment(start=start, end=end, text=text))

    segments = _dedupe(segments)

    return segments


def _ts_to_seconds(h: int, m: int, s: int, ms: int) -> float:
    """Convert timestamp components to float seconds."""
    return h * 3600 + m * 60 + s + ms / 1000.0


def _dedupe(segments: list[CaptionSegment]) -> list[CaptionSegment]:
    """Deduplicate YouTube auto-caption rolling duplicates.

    YouTube auto-subs often have each line repeated 2-3 times as it scrolls.
    This detects and removes those duplicates by comparing consecutive segments
    that share text content.
    """
    if not segments:
        return segments

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        # If this segment's text is a substring of the previous (or vice versa),
        # and they overlap in time, skip it
        if (seg.text in prev.text or prev.text in seg.text) and seg.start < prev.end + 0.5:
            if len(seg.text) > len(prev.text):
                result[-1] = seg
        else:
            result.append(seg)

    return result


def filter_range(
    segments: list[CaptionSegment],
    start: float | None = None,
    end: float | None = None,
) -> list[CaptionSegment]:
    """Filter caption segments to a time range."""
    result = []
    for seg in segments:
        # Include segment if it overlaps with the range
        if start is not None and seg.end < start:
            continue
        if end is not None and seg.start > end:
            continue
        result.append(seg)
    return result


def format_transcript(segments: list[CaptionSegment]) -> str:
    """Format caption segments into readable transcript text."""
    lines = []
    for seg in segments:
        ts = _format_ts(seg.start)
        lines.append(f"[{ts}] {seg.text}")
    return '\n'.join(lines)


def _format_ts(seconds: float) -> str:
    """Format seconds to MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
