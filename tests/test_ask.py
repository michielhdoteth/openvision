"""Tests for ask command."""
import pytest
from typer.testing import CliRunner
from cli.app import app
from openvision.models import AskResult
from cli.commands.ask import _is_uncertain

runner = CliRunner()


class TestAskModels:
    def test_ask_result_creation(self):
        result = AskResult(
            answer="Yes, there is a red car.",
            evidence=[
                {"timestamp": "00:05", "time_seconds": 5.0, "answer": "A red car is visible", "confidence": "medium"},
            ],
            confidence="high",
        )
        assert "red car" in result.answer
        assert len(result.evidence) == 1


class TestIsUncertain:
    def test_uncertain_phrases(self):
        assert _is_uncertain("I cannot determine what is in the image")
        assert _is_uncertain("There is no visible object matching that description")
        assert _is_uncertain("It is not clear from this frame")

    def test_certain_responses(self):
        assert not _is_uncertain("The image shows a red car on a road")
        assert not _is_uncertain("A person is walking toward the door")
        assert not _is_uncertain("There is a gas cylinder near the wall")


class TestAskCli:
    def test_ask_nonexistent(self):
        result = runner.invoke(app, ["ask", "nonexistent.mp4", "what is this?"])
        assert result.exit_code != 0

    def test_ask_empty_question(self):
        result = runner.invoke(app, ["ask", "test.mp4", ""])
        assert result.exit_code != 0

    def test_ask_missing_lmstudio(self):
        """Without LM Studio, should fail gracefully."""
        from pathlib import Path
        videos = list(Path("test_data").glob("*.mp4")) if Path("test_data").exists() else []
        if not videos:
            pytest.skip("No test video available")

        result = runner.invoke(app, ["ask", str(videos[0]), "Is there a person?"])
        assert result.exit_code in (0, 1)  # Either works or fails cleanly
def test_uncertain_includes_provider_failures():
    from cli.commands.ask import _is_uncertain

    assert _is_uncertain("Request to lmstudio timed out after 2s.")
    assert _is_uncertain("Could not connect to lmstudio")
