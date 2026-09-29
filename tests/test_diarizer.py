from providers.diarizer import SpeakerSegment, merge_transcript_with_diarization
from providers.parakeet import TranscriptSegment


def test_transcript_segment_uses_speaker_with_greatest_time_overlap():
    transcript = [TranscriptSegment(1.0, 3.0, "hello", 0.9)]
    diarization = [
        SpeakerSegment(1.0, 2.4, "SPEAKER_00"),
        SpeakerSegment(2.4, 3.0, "SPEAKER_01"),
    ]

    merged = merge_transcript_with_diarization(transcript, diarization)

    assert merged[0].speaker == "SPEAKER_00"
    assert merged[0].text == "hello"
