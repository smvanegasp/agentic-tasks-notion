from unittest.mock import MagicMock, patch


@patch("agentic_tasks.agent.transcribe.OpenAI")
def test_transcribe_returns_text(mock_openai_cls):
    from agentic_tasks.agent.transcribe import transcribe_audio

    response = MagicMock()
    response.text = "hola mundo"

    client = MagicMock()
    client.audio.transcriptions.create.return_value = response
    mock_openai_cls.return_value = client

    result = transcribe_audio(b"fake-audio")

    assert result == "hola mundo"


@patch("agentic_tasks.agent.transcribe.OpenAI")
def test_transcribe_uses_configured_model(mock_openai_cls):
    from agentic_tasks.agent.transcribe import transcribe_audio
    from agentic_tasks.config import get_settings

    response = MagicMock()
    response.text = "x"
    client = MagicMock()
    client.audio.transcriptions.create.return_value = response
    mock_openai_cls.return_value = client

    transcribe_audio(b"fake-audio")

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == get_settings().transcription_model


@patch("agentic_tasks.agent.transcribe.OpenAI")
def test_transcribe_handles_none_text(mock_openai_cls):
    """If the API returns text=None, we should return empty string, not crash."""
    from agentic_tasks.agent.transcribe import transcribe_audio

    response = MagicMock()
    response.text = None
    client = MagicMock()
    client.audio.transcriptions.create.return_value = response
    mock_openai_cls.return_value = client

    assert transcribe_audio(b"fake") == ""
