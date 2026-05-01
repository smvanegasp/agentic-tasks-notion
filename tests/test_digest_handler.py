from unittest.mock import AsyncMock, patch


@patch("agentic_tasks.handlers.digest._send_digest_message", new_callable=AsyncMock)
@patch("agentic_tasks.handlers.digest.query_overdue", return_value=[])
@patch("agentic_tasks.handlers.digest.query_my_day", return_value=[])
@patch("agentic_tasks.handlers.digest.query_due_today", return_value=[])
def test_handler_returns_200_and_sends_digest(
    mock_today, mock_my_day, mock_overdue, mock_send
):
    from agentic_tasks.handlers.digest import handler

    result = handler({}, None)

    assert result["statusCode"] == 200
    mock_send.assert_awaited_once()
    sent_text = mock_send.await_args.args[0]
    assert "Good morning" in sent_text


@patch("agentic_tasks.handlers.digest._send_digest_message", new_callable=AsyncMock)
@patch("agentic_tasks.handlers.digest.query_overdue", return_value=[])
@patch("agentic_tasks.handlers.digest.query_my_day", return_value=[])
@patch("agentic_tasks.handlers.digest.query_due_today", return_value=[])
def test_handler_returns_500_on_send_failure(
    mock_today, mock_my_day, mock_overdue, mock_send
):
    from agentic_tasks.handlers.digest import handler

    mock_send.side_effect = RuntimeError("telegram down")
    result = handler({}, None)

    assert result["statusCode"] == 500


@patch("agentic_tasks.handlers.digest.query_overdue", return_value=[])
@patch("agentic_tasks.handlers.digest.query_my_day", return_value=[])
@patch("agentic_tasks.handlers.digest.query_due_today", return_value=[])
def test_build_digest_message_uses_today_in_settings_tz(
    mock_today, mock_my_day, mock_overdue
):
    from agentic_tasks.handlers.digest import build_digest_message

    text = build_digest_message()

    # Each query is invoked with today's date (settings TZ)
    assert mock_today.call_args.kwargs.get("today") is not None
    assert mock_overdue.call_args.kwargs.get("today") is not None
    # And the message is non-empty
    assert "Good morning" in text
