from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import persistence
from persistence import (
    append_audit_event,
    build_audit_event,
    build_config_version_record,
    compare_config_versions,
    compute_file_hash,
    delete_json_record,
    ensure_storage_dirs,
    list_config_versions,
    list_json_records,
    load_config_version,
    load_json_record,
    make_record_id,
    read_audit_events,
    save_config_version,
    save_json_record,
    validate_audit_event,
    validate_record_id,
)


def test_make_record_id_is_deterministic() -> None:
    assert make_record_id("Busy Season Week 1") == "busy_season_week_1"
    assert make_record_id("Busy Season Week 1") == "busy_season_week_1"


def test_blank_record_name_fails() -> None:
    with pytest.raises(ValueError, match="record name cannot be blank"):
        make_record_id("  ")


def test_unsafe_record_id_fails() -> None:
    with pytest.raises(ValueError, match="record_id cannot be blank"):
        validate_record_id("")
    with pytest.raises(ValueError, match="path separators|path traversal"):
        validate_record_id("../escape")
    with pytest.raises(ValueError, match="lowercase letters"):
        validate_record_id("Bad-ID")


def test_ensure_storage_dirs_creates_only_allowed_storage_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(persistence, "SCENARIO_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(persistence, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setattr(persistence, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(persistence, "CONFIG_VERSION_DIR", tmp_path / "config_versions")

    paths = ensure_storage_dirs()

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "audit",
        "config_versions",
        "history",
        "scenarios",
    ]
    assert set(paths) == {"scenarios", "history", "audit", "config_versions"}


def test_save_json_record_writes_deterministic_json(tmp_path: Path) -> None:
    saved = save_json_record(tmp_path, "record_a", {"b": 2, "a": 1})
    assert saved == {"b": 2, "a": 1}
    assert (tmp_path / "record_a.json").read_text(encoding="utf-8") == (
        '{\n  "a": 1,\n  "b": 2\n}\n'
    )


def test_duplicate_save_without_overwrite_fails(tmp_path: Path) -> None:
    save_json_record(tmp_path, "record_a", {"a": 1})
    with pytest.raises(ValueError, match="already exists"):
        save_json_record(tmp_path, "record_a", {"a": 2})


def test_duplicate_save_with_overwrite_succeeds(tmp_path: Path) -> None:
    save_json_record(tmp_path, "record_a", {"a": 1})
    saved = save_json_record(tmp_path, "record_a", {"a": 2}, overwrite=True)
    assert saved == {"a": 2}
    assert load_json_record(tmp_path, "record_a") == {"a": 2}


def test_load_json_record_returns_saved_record(tmp_path: Path) -> None:
    save_json_record(tmp_path, "record_a", {"a": 1})
    assert load_json_record(tmp_path, "record_a") == {"a": 1}


def test_list_json_records_returns_stable_sorted_records(tmp_path: Path) -> None:
    save_json_record(tmp_path, "b_record", {"id": "b"})
    save_json_record(tmp_path, "a_record", {"id": "a"})
    assert [row["id"] for row in list_json_records(tmp_path)] == ["a", "b"]


def test_malformed_json_record_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{bad-json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_json_record(tmp_path, "bad")
    with pytest.raises(ValueError, match="malformed JSON"):
        list_json_records(tmp_path)


def test_delete_json_record_deletes_exact_record_only(tmp_path: Path) -> None:
    save_json_record(tmp_path, "record_a", {"a": 1})
    save_json_record(tmp_path, "record_b", {"b": 2})
    assert delete_json_record(tmp_path, "record_a") is True
    assert not (tmp_path / "record_a.json").exists()
    assert (tmp_path / "record_b.json").exists()
    assert delete_json_record(tmp_path, "record_a") is False


def test_path_traversal_record_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path separators|path traversal"):
        save_json_record(tmp_path, "../escape", {"a": 1})


def test_build_audit_event_creates_required_schema() -> None:
    event = build_audit_event(
        "scenario_saved",
        "scenario",
        "scenario_a",
        {"source": "test"},
        created_at="2026-05-13T00:00:00+00:00",
    )
    assert set(event) == {
        "event_id",
        "created_at",
        "event_type",
        "target_type",
        "target_id",
        "metadata",
    }
    assert event["event_type"] == "scenario_saved"
    assert event["metadata"] == {"source": "test"}


def test_audit_validation_catches_missing_required_keys() -> None:
    event = build_audit_event(
        "scenario_saved",
        "scenario",
        "scenario_a",
        created_at="2026-05-13T00:00:00+00:00",
    )
    event.pop("metadata")
    with pytest.raises(ValueError, match="missing required key"):
        validate_audit_event(event)


def test_append_audit_event_appends_jsonl(tmp_path: Path) -> None:
    audit_file = tmp_path / "audit_log.jsonl"
    first = build_audit_event(
        "scenario_saved",
        "scenario",
        "a",
        created_at="2026-05-13T00:00:00+00:00",
    )
    second = build_audit_event(
        "scenario_deleted",
        "scenario",
        "a",
        created_at="2026-05-13T00:00:01+00:00",
    )

    append_audit_event(first, audit_file=audit_file)
    append_audit_event(second, audit_file=audit_file)

    lines = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "scenario_saved"
    assert json.loads(lines[1])["event_type"] == "scenario_deleted"


def test_read_audit_events_returns_appended_events_in_order(tmp_path: Path) -> None:
    audit_file = tmp_path / "audit_log.jsonl"
    events = [
        build_audit_event("one", "target", "a", created_at="2026-05-13T00:00:00+00:00"),
        build_audit_event("two", "target", "b", created_at="2026-05-13T00:00:01+00:00"),
    ]
    for event in events:
        append_audit_event(event, audit_file=audit_file)
    assert [event["event_type"] for event in read_audit_events(audit_file=audit_file)] == [
        "one",
        "two",
    ]


def test_malformed_audit_jsonl_fails_clearly(tmp_path: Path) -> None:
    audit_file = tmp_path / "audit_log.jsonl"
    audit_file.write_text("{bad-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Audit log line 1 contains malformed JSON"):
        read_audit_events(audit_file=audit_file)


def test_compute_file_hash_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    assert compute_file_hash(source) == compute_file_hash(source)


def test_compute_file_hash_fails_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        compute_file_hash(tmp_path / "missing.csv")


def test_build_config_version_record_stores_hashes_only_not_csv_contents(tmp_path: Path) -> None:
    source = tmp_path / "class_config.csv"
    source.write_text("secret,csv,content\n", encoding="utf-8")
    record = build_config_version_record(
        config_version_name="Baseline",
        source_paths={"class_config": str(source)},
        created_at="2026-05-13T00:00:00+00:00",
    )

    serialized = json.dumps(record, sort_keys=True)
    assert "secret,csv,content" not in serialized
    assert record["source_paths"] == {"class_config": str(source)}
    assert set(record["file_hashes"]) == {"class_config"}


def test_save_load_list_config_version_works(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a\n", encoding="utf-8")
    storage_dir = tmp_path / "config_versions"
    audit_file = tmp_path / "audit" / "audit_log.jsonl"

    saved = save_config_version(
        config_version_name="Version B",
        source_paths={"source": str(source)},
        storage_dir=storage_dir,
        audit_file=audit_file,
    )
    save_config_version(
        config_version_name="Version A",
        source_paths={"source": str(source)},
        storage_dir=storage_dir,
        audit_file=audit_file,
    )

    assert load_config_version(saved["config_version_id"], storage_dir=storage_dir) == saved
    assert [row["config_version_id"] for row in list_config_versions(storage_dir=storage_dir)] == [
        "version_a",
        "version_b",
    ]
    assert read_audit_events(audit_file=audit_file)[0]["event_type"] == "config_version_saved"


def test_compare_config_versions_detects_unchanged_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a\n", encoding="utf-8")
    storage_dir = tmp_path / "config_versions"
    audit_file = tmp_path / "audit_log.jsonl"
    save_config_version(
        config_version_name="Version 1",
        source_paths={"source": str(source)},
        storage_dir=storage_dir,
        audit_file=audit_file,
    )
    save_config_version(
        config_version_name="Version 2",
        source_paths={"source": str(source)},
        storage_dir=storage_dir,
        audit_file=audit_file,
    )

    comparison = compare_config_versions("version_1", "version_2", storage_dir=storage_dir)
    assert comparison["source_comparisons"][0]["change_type"] == "unchanged"


def test_compare_config_versions_detects_changed_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a\n", encoding="utf-8")
    storage_dir = tmp_path / "config_versions"
    audit_file = tmp_path / "audit_log.jsonl"
    save_config_version(
        config_version_name="Version 1",
        source_paths={"source": str(source)},
        storage_dir=storage_dir,
        audit_file=audit_file,
    )
    source.write_text("b\n", encoding="utf-8")
    save_config_version(
        config_version_name="Version 2",
        source_paths={"source": str(source)},
        storage_dir=storage_dir,
        audit_file=audit_file,
    )

    comparison = compare_config_versions("version_1", "version_2", storage_dir=storage_dir)
    assert comparison["source_comparisons"][0]["change_type"] == "changed"


def test_persistence_helpers_do_not_mutate_input_records(tmp_path: Path) -> None:
    record = {"nested": {"b": 2}}
    before = copy.deepcopy(record)
    save_json_record(tmp_path, "record_a", record)
    assert record == before

    event = build_audit_event(
        "scenario_saved",
        "scenario",
        "scenario_a",
        {"nested": {"a": 1}},
        created_at="2026-05-13T00:00:00+00:00",
    )
    event_before = copy.deepcopy(event)
    append_audit_event(event, audit_file=tmp_path / "audit.jsonl")
    assert event == event_before


def test_persistence_layer_does_not_write_to_data_or_config(tmp_path: Path) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    config_before = sorted(path.name for path in (base_dir / "config").iterdir())
    data_before = sorted(path.name for path in (base_dir / "data").iterdir())
    source = tmp_path / "source.csv"
    source.write_text("a\n", encoding="utf-8")

    save_json_record(tmp_path / "records", "record_a", {"a": 1})
    save_config_version(
        config_version_name="Version 1",
        source_paths={"source": str(source)},
        storage_dir=tmp_path / "config_versions",
        audit_file=tmp_path / "audit_log.jsonl",
    )

    assert sorted(path.name for path in (base_dir / "config").iterdir()) == config_before
    assert sorted(path.name for path in (base_dir / "data").iterdir()) == data_before
