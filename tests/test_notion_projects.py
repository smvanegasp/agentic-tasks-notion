from unittest.mock import MagicMock, patch

from agentic_tasks.notion_io.projects import find_project_by_name, list_projects


def _project_page(name: str, page_id: str) -> dict:
    return {
        "id": page_id,
        "properties": {"Name": {"title": [{"plain_text": name}]}},
    }


def _schema_with_title() -> dict:
    return {"Name": {"type": "title"}}


@patch("agentic_tasks.notion_io.projects.get_projects_data_source_id", return_value="ds-proj")
@patch("agentic_tasks.notion_io.projects.get_client")
@patch("agentic_tasks.notion_io.projects.get_projects_db_schema")
def test_list_projects_paginates(mock_schema, mock_get_client, _mock_ds_id):
    mock_schema.return_value = _schema_with_title()
    mock_client = MagicMock()
    mock_client.data_sources.query.side_effect = [
        {
            "results": [_project_page("Alpha", "p1")],
            "has_more": True,
            "next_cursor": "cursor-1",
        },
        {"results": [_project_page("Beta", "p2")], "has_more": False},
    ]
    mock_get_client.return_value = mock_client

    projects = list_projects()

    assert [p.name for p in projects] == ["Alpha", "Beta"]
    assert mock_client.data_sources.query.call_count == 2
    second_call = mock_client.data_sources.query.call_args_list[1].kwargs
    assert second_call["start_cursor"] == "cursor-1"
    assert second_call["data_source_id"] == "ds-proj"


@patch("agentic_tasks.notion_io.projects.get_projects_data_source_id", return_value="ds-proj")
@patch("agentic_tasks.notion_io.projects.get_client")
@patch("agentic_tasks.notion_io.projects.get_projects_db_schema")
def test_find_project_fuzzy_matches(mock_schema, mock_get_client, _mock_ds_id):
    mock_schema.return_value = _schema_with_title()
    mock_client = MagicMock()
    mock_client.data_sources.query.return_value = {
        "results": [
            _project_page("Breaking Benjamin tribute", "p1"),
            _project_page("Solar car project", "p2"),
        ],
        "has_more": False,
    }
    mock_get_client.return_value = mock_client

    p = find_project_by_name("breaking benjamin")

    assert p is not None
    assert p.name == "Breaking Benjamin tribute"


@patch("agentic_tasks.notion_io.projects.get_projects_data_source_id", return_value="ds-proj")
@patch("agentic_tasks.notion_io.projects.get_client")
@patch("agentic_tasks.notion_io.projects.get_projects_db_schema")
def test_find_project_returns_none_below_cutoff(mock_schema, mock_get_client, _mock_ds_id):
    mock_schema.return_value = _schema_with_title()
    mock_client = MagicMock()
    mock_client.data_sources.query.return_value = {
        "results": [_project_page("Alpha", "p1")],
        "has_more": False,
    }
    mock_get_client.return_value = mock_client

    p = find_project_by_name("zzzzzzz", score_cutoff=80)

    assert p is None


@patch("agentic_tasks.notion_io.projects.get_projects_data_source_id", return_value="ds-proj")
@patch("agentic_tasks.notion_io.projects.get_client")
@patch("agentic_tasks.notion_io.projects.get_projects_db_schema")
def test_find_project_uses_detected_title_property(mock_schema, mock_get_client, _mock_ds_id):
    mock_schema.return_value = {
        "ProjectName": {"type": "title"},
        "Status": {"type": "status"},
    }
    mock_client = MagicMock()
    mock_client.data_sources.query.return_value = {
        "results": [
            {
                "id": "p1",
                "properties": {"ProjectName": {"title": [{"plain_text": "Custom"}]}},
            }
        ],
        "has_more": False,
    }
    mock_get_client.return_value = mock_client

    p = find_project_by_name("Custom")

    assert p is not None
    assert p.name == "Custom"
