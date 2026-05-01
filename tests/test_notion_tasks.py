from datetime import date
from unittest.mock import MagicMock, patch

from agentic_tasks.notion_io.schema import Priority, Status
from agentic_tasks.notion_io.tasks import (
    complete_task,
    create_task,
    query_tasks,
    query_today,
    update_task,
)


def _fake_page(
    name="Test",
    status="To Do",
    priority="High",
    due="2026-04-30",
    my_day=False,
):
    return {
        "id": "page-id-1",
        "url": "https://notion.so/page-1",
        "properties": {
            "Name": {"title": [{"plain_text": name}]},
            "Status": {"status": {"name": status}},
            "Priority": {"status": {"name": priority}},
            "Due": {"date": {"start": due} if due else None},
            "Project": {"relation": [{"id": "proj-1"}]},
            "Description": {"rich_text": [{"plain_text": "details"}]},
            "Labels": {"multi_select": [{"name": "urgent"}]},
            "My Day": {"checkbox": my_day},
        },
    }


@patch("agentic_tasks.notion_io.tasks.get_tasks_data_source_id", return_value="ds-tasks")
@patch("agentic_tasks.notion_io.tasks.get_client")
def test_parse_preserves_spanish_characters(mock_get_client, _mock_ds_id):
    spanish_name = "REL: Llamar a mamá el día de la madre — café con piñón"
    spanish_desc = "¿Confirmaste la reservación? Sí, ¡todo listo!"
    mock_client = MagicMock()
    mock_client.data_sources.query.return_value = {
        "results": [_fake_page(name=spanish_name)],
    }
    # Override description in the fake page
    mock_client.data_sources.query.return_value["results"][0]["properties"][
        "Description"
    ] = {"rich_text": [{"plain_text": spanish_desc}]}
    mock_get_client.return_value = mock_client

    tasks = query_tasks()

    assert len(tasks) == 1
    assert tasks[0].name == spanish_name
    assert tasks[0].description == spanish_desc


@patch("agentic_tasks.notion_io.tasks.get_tasks_data_source_id", return_value="ds-tasks")
@patch("agentic_tasks.notion_io.tasks.get_client")
def test_create_preserves_spanish_characters(mock_get_client, _mock_ds_id):
    spanish_name = "Día de la Madre — comprar tarjeta"
    spanish_desc = "Pasar por la tienda de la esquina, ¿sí?"
    mock_client = MagicMock()
    mock_client.pages.create.return_value = _fake_page(name=spanish_name)
    mock_get_client.return_value = mock_client

    create_task(name=spanish_name, description=spanish_desc)

    props = mock_client.pages.create.call_args.kwargs["properties"]
    assert props["Name"]["title"][0]["text"]["content"] == spanish_name
    assert props["Description"]["rich_text"][0]["text"]["content"] == spanish_desc


@patch("agentic_tasks.notion_io.tasks.get_tasks_data_source_id", return_value="ds-tasks")
@patch("agentic_tasks.notion_io.tasks.get_client")
def test_query_tasks_parses_pages(mock_get_client, _mock_ds_id):
    mock_client = MagicMock()
    mock_client.data_sources.query.return_value = {"results": [_fake_page()]}
    mock_get_client.return_value = mock_client

    tasks = query_tasks()

    assert len(tasks) == 1
    t = tasks[0]
    assert t.name == "Test"
    assert t.status == "To Do"
    assert t.priority == "High"
    assert t.due == date(2026, 4, 30)
    assert t.project_ids == ["proj-1"]
    assert t.labels == ["urgent"]
    assert t.description == "details"
    assert t.my_day is False


@patch("agentic_tasks.notion_io.tasks.get_tasks_data_source_id", return_value="ds-tasks")
@patch("agentic_tasks.notion_io.tasks.get_client")
def test_query_today_filter_shape(mock_get_client, _mock_ds_id):
    mock_client = MagicMock()
    mock_client.data_sources.query.return_value = {"results": []}
    mock_get_client.return_value = mock_client

    query_today(today=date(2026, 4, 30))

    call_kwargs = mock_client.data_sources.query.call_args.kwargs
    f = call_kwargs["filter"]
    assert "and" in f
    or_clause = next(c["or"] for c in f["and"] if "or" in c)
    assert any(
        "date" in c and c["date"].get("equals") == "2026-04-30" for c in or_clause
    )
    assert any(
        "checkbox" in c and c["checkbox"].get("equals") is True for c in or_clause
    )


@patch("agentic_tasks.notion_io.tasks.get_tasks_data_source_id", return_value="ds-tasks")
@patch("agentic_tasks.notion_io.tasks.get_client")
def test_create_task_builds_properties(mock_get_client, _mock_ds_id):
    mock_client = MagicMock()
    mock_client.pages.create.return_value = _fake_page(name="New Task")
    mock_get_client.return_value = mock_client

    create_task(
        name="New Task",
        priority=Priority.HIGH,
        due=date(2026, 5, 1),
        project_ids=["proj-1"],
        my_day=True,
        description="some details",
    )

    create_kwargs = mock_client.pages.create.call_args.kwargs
    assert create_kwargs["parent"] == {"data_source_id": "ds-tasks"}
    props = create_kwargs["properties"]
    assert props["Name"]["title"][0]["text"]["content"] == "New Task"
    assert props["Status"]["status"]["name"] == "To Do"
    assert props["Priority"]["status"]["name"] == "High"
    assert props["Due"]["date"]["start"] == "2026-05-01"
    assert props["Project"]["relation"][0]["id"] == "proj-1"
    assert props["My Day"]["checkbox"] is True
    assert props["Description"]["rich_text"][0]["text"]["content"] == "some details"


@patch("agentic_tasks.notion_io.tasks.get_client")
def test_complete_task_sets_status_done(mock_get_client):
    mock_client = MagicMock()
    mock_client.pages.update.return_value = _fake_page(status="Done")
    mock_get_client.return_value = mock_client

    complete_task("page-id-1")

    props = mock_client.pages.update.call_args.kwargs["properties"]
    assert props["Status"]["status"]["name"] == "Done"


@patch("agentic_tasks.notion_io.tasks.get_client")
def test_update_task_only_includes_passed_fields(mock_get_client):
    mock_client = MagicMock()
    mock_client.pages.update.return_value = _fake_page()
    mock_get_client.return_value = mock_client

    update_task("page-id-1", priority=Priority.LOW)

    props = mock_client.pages.update.call_args.kwargs["properties"]
    assert "Priority" in props
    assert props["Priority"]["status"]["name"] == "Low"
    assert "Status" not in props
    assert "Name" not in props
    assert "Due" not in props


@patch("agentic_tasks.notion_io.tasks.get_client")
def test_update_task_clears_due_when_none(mock_get_client):
    mock_client = MagicMock()
    mock_client.pages.update.return_value = _fake_page(due=None)
    mock_get_client.return_value = mock_client

    update_task("page-id-1", due=None)

    props = mock_client.pages.update.call_args.kwargs["properties"]
    assert props["Due"] == {"date": None}


@patch("agentic_tasks.notion_io.tasks.get_client")
def test_set_status_via_update(mock_get_client):
    mock_client = MagicMock()
    mock_client.pages.update.return_value = _fake_page(status="Doing")
    mock_get_client.return_value = mock_client

    update_task("page-id-1", status=Status.DOING)

    props = mock_client.pages.update.call_args.kwargs["properties"]
    assert props["Status"]["status"]["name"] == "Doing"
