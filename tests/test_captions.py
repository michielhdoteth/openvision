"""Tests for VTT caption parsing and deduplication."""
import pytest
from openvision.core.captions import (
    CaptionSegment,
    parse_vtt,
    filter_range,
    format_transcript,
    _dedupe,
    _ts_to_seconds,
    _format_ts,
)


SAMPLE_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.500 --> 00:00:06.000
This is a test

00:00:06.500 --> 00:00:09.000
Of caption parsing
"""

SAMPLE_VTT_WITH_HOURS = """WEBVTT

01:02:03.400 --> 01:02:05.800
Long video segment

01:30:00.000 --> 01:30:02.500
Mid-roll caption
"""

SAMPLE_VTT_TAGS = """WEBVTT

00:00:01.000 --> 00:00:03.000
<c>Hello</c> <c>world</c>

00:00:03.500 --> 00:00:06.000
This <i>is</i> a <b>test</b>
"""

SAMPLE_VTT_NOTES = """WEBVTT

NOTE This is a comment

00:00:01.000 --> 00:00:03.000
Hello world

NOTE Another comment
"""

SAMPLE_VTT_YOUTUBE_DEDUP = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world

00:00:01.500 --> 00:00:03.000
Hello world how are you

00:00:02.800 --> 00:00:04.000
Hello world how are you doing today
"""

SAMPLE_VTT_EMPTY = """WEBVTT

"""

SAMPLE_VTT_SHORT_TS = """WEBVTT

01:30.500 --> 02:00.000
Short format caption
"""


class TestCaptionSegment:
    def test_dataclass_fields(self):
        seg = CaptionSegment(start=1.0, end=2.5, text="hello")
        assert seg.start == 1.0
        assert seg.end == 2.5
        assert seg.text == "hello"

    def test_equality(self):
        a = CaptionSegment(start=1.0, end=2.0, text="x")
        b = CaptionSegment(start=1.0, end=2.0, text="x")
        assert a == b


class TestTsToSeconds:
    def test_zero(self):
        assert _ts_to_seconds(0, 0, 0, 0) == 0.0

    def test_hours_minutes_seconds_ms(self):
        assert _ts_to_seconds(1, 2, 3, 400) == 3723.4

    def test_minutes_only(self):
        assert _ts_to_seconds(0, 5, 30, 0) == 330.0

    def test_seconds_only(self):
        assert _ts_to_seconds(0, 0, 45, 500) == 45.5


class TestFormatTs:
    def test_zero(self):
        assert _format_ts(0) == "00:00"

    def test_minutes(self):
        assert _format_ts(125) == "02:05"

    def test_large(self):
        assert _format_ts(3661) == "61:01"


class TestParseVtt:
    def test_basic_parse(self):
        segments = parse_vtt(SAMPLE_VTT)
        assert len(segments) == 3
        assert segments[0].text == "Hello world"
        assert segments[0].start == 1.0
        assert segments[0].end == 3.0
        assert segments[1].text == "This is a test"
        assert segments[2].text == "Of caption parsing"

    def test_with_hours(self):
        segments = parse_vtt(SAMPLE_VTT_WITH_HOURS)
        assert len(segments) == 2
        assert segments[0].start == pytest.approx(3723.4)
        assert segments[0].end == pytest.approx(3725.8)
        assert segments[1].start == pytest.approx(5400.0)
        assert segments[1].end == pytest.approx(5402.5)

    def test_strips_html_tags(self):
        segments = parse_vtt(SAMPLE_VTT_TAGS)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[1].text == "This is a test"

    def test_notes_ignored(self):
        segments = parse_vtt(SAMPLE_VTT_NOTES)
        assert len(segments) == 1
        assert segments[0].text == "Hello world"

    def test_empty_vtt(self):
        segments = parse_vtt(SAMPLE_VTT_EMPTY)
        assert len(segments) == 0

    def test_empty_string(self):
        segments = parse_vtt("")
        assert len(segments) == 0

    def test_youtube_dedup(self):
        segments = parse_vtt(SAMPLE_VTT_YOUTUBE_DEDUP)
        # Should deduplicate rolling subs - first two overlap
        assert len(segments) < 3
        # The longest version should survive
        assert any("doing today" in s.text for s in segments)


class TestDedupe:
    def test_empty(self):
        assert _dedupe([]) == []

    def test_single_segment(self):
        seg = CaptionSegment(start=0, end=1, text="hello")
        assert _dedupe([seg]) == [seg]

    def test_no_duplicates(self):
        segs = [
            CaptionSegment(start=0, end=1, text="hello"),
            CaptionSegment(start=2, end=3, text="world"),
        ]
        result = _dedupe(segs)
        assert len(result) == 2

    def test_exact_duplicate(self):
        segs = [
            CaptionSegment(start=0, end=1, text="hello"),
            CaptionSegment(start=0.5, end=1.5, text="hello"),
        ]
        result = _dedupe(segs)
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_overlapping_substring(self):
        """When seg text is substring of prev, and they overlap, dedup."""
        segs = [
            CaptionSegment(start=0, end=2, text="Hello world"),
            CaptionSegment(start=1.5, end=3, text="Hello"),
        ]
        result = _dedupe(segs)
        assert len(result) == 1
        assert result[0].text == "Hello world"

    def test_overlapping_superset(self):
        """When seg has longer text and overlaps prev, keep seg."""
        segs = [
            CaptionSegment(start=0, end=2, text="Hello"),
            CaptionSegment(start=1.5, end=3, text="Hello world"),
        ]
        result = _dedupe(segs)
        assert len(result) == 1
        assert result[0].text == "Hello world"

    def test_non_overlapping_kept(self):
        """Non-overlapping segments with same text are kept."""
        segs = [
            CaptionSegment(start=0, end=1, text="hello"),
            CaptionSegment(start=5, end=6, text="hello"),
        ]
        result = _dedupe(segs)
        assert len(result) == 2

    def test_borderline_timing(self):
        """Segments exactly at the 0.5s boundary are kept separate."""
        segs = [
            CaptionSegment(start=0, end=1, text="hello"),
            CaptionSegment(start=1.6, end=2, text="hello"),
        ]
        result = _dedupe(segs)
        assert len(result) == 2


class TestFilterRange:
    def test_no_filter(self):
        segs = [
            CaptionSegment(start=0, end=1, text="a"),
            CaptionSegment(start=2, end=3, text="b"),
        ]
        assert len(filter_range(segs)) == 2

    def test_start_filter(self):
        segs = [
            CaptionSegment(start=0, end=1, text="a"),
            CaptionSegment(start=2, end=3, text="b"),
            CaptionSegment(start=4, end=5, text="c"),
        ]
        result = filter_range(segs, start=1.5)
        assert len(result) == 2
        assert result[0].text == "b"
        assert result[1].text == "c"

    def test_end_filter(self):
        segs = [
            CaptionSegment(start=0, end=1, text="a"),
            CaptionSegment(start=2, end=3, text="b"),
            CaptionSegment(start=4, end=5, text="c"),
        ]
        result = filter_range(segs, end=3.5)
        assert len(result) == 2
        assert result[0].text == "a"
        assert result[1].text == "b"

    def test_both_filters(self):
        segs = [
            CaptionSegment(start=0, end=1, text="a"),
            CaptionSegment(start=2, end=3, text="b"),
            CaptionSegment(start=4, end=5, text="c"),
            CaptionSegment(start=6, end=7, text="d"),
        ]
        result = filter_range(segs, start=1.5, end=5.5)
        assert len(result) == 2
        assert result[0].text == "b"
        assert result[1].text == "c"

    def test_overlapping_segment_included(self):
        """Segment that partially overlaps range is included."""
        segs = [CaptionSegment(start=0, end=3, text="spanning")]
        result = filter_range(segs, start=2, end=4)
        assert len(result) == 1

    def test_empty(self):
        assert len(filter_range([], start=0, end=10)) == 0


class TestFormatTranscript:
    def test_basic(self):
        segs = [
            CaptionSegment(start=0, end=1, text="Hello"),
            CaptionSegment(start=65, end=70, text="World"),
        ]
        result = format_transcript(segs)
        assert "[00:00] Hello" in result
        assert "[01:05] World" in result
        assert result.count('\n') == 1

    def test_empty(self):
        assert format_transcript([]) == ""


class TestParseVttEdgeCases:
    def test_malformed_timestamp(self):
        """Lines without valid timestamps are skipped."""
        vtt = """WEBVTT

This line has no timestamp
And neither does this

00:00:01.000 --> 00:00:03.000
Valid segment
"""
        segments = parse_vtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "Valid segment"

    def test_multiple_text_lines(self):
        """Multiple text lines under one timestamp are joined."""
        vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Line one
Line two
Line three
"""
        segments = parse_vtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "Line one Line two Line three"

    def test_comma_separator(self):
        """Comma is accepted as timestamp separator."""
        vtt = """WEBVTT

00:00:01,000 --> 00:00:03,000
Comma format
"""
        segments = parse_vtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "Comma format"
