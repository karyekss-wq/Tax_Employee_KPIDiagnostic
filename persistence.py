from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = BASE_DIR / "scenarios"
HISTORY_DIR = BASE_DIR / "history"
AUDIT_DIR = BASE_DIR / "audit"
CONFIG_VERSION_DIR = BASE_DIR / "config_versions"
AUDIT_LOG_FILE = AUDIT_DIR / "audit_log.jsonl"

SAFE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")

AUDIT_EVENT_KEYS = [
    "event_id",
    "created_at",
    "event_type",
    "target_type",
    "target_id",
    "metadata",
]

CONFIG_VERSION_KEYS = [
    "config_version_id",
    "config_version_name",
    "created_at",
    "source_paths",
    "file_hashes",
    "notes",
]

DEFAULT_CONFIG_SOURCE_PATHS = {
    "class_config": "config/class_config.csv",
    "adjustment_config": "config/adjustment_config.csv",
    "tasks": "data/tasks.csv",
    "flags": "data/flags.csv",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty ISO timestamp string.")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO timestamp string.") from exc
    return value


def get_storage_paths() -> dict[str, Path]:
    return {
        "scenarios": SCENARIO_DIR,
        "history": HISTORY_DIR,
        "audit": AUDIT_DIR,
        "config_versions": CONFIG_VERSION_DIR,
    }


def ensure_storage_dirs() -> dict[str, Path]:
    paths = get_storage_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_record_id(name: str) -> str:
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("record name cannot be blank.")
    record_id = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    record_id = re.sub(r"_+", "_", record_id).strip("_")
    if record_id == "":
        raise ValueError("record name collapses to an empty record_id.")
    return record_id


def validate_record_id(record_id: str) -> str:
    if not isinstance(record_id, str) or record_id.strip() == "":
        raise ValueError("record_id cannot be blank.")
    if record_id != record_id.strip():
        raise ValueError("record_id must not contain leading or trailing whitespace.")
    if "/" in record_id or "\\" in record_id:
        raise ValueError("record_id must not contain path separators.")
    if record_id in {".", ".."} or ".." in record_id:
        raise ValueError("record_id must not contain path traversal.")
    if not SAFE_ID_PATTERN.fullmatch(record_id):
        raise ValueError("record_id must contain only lowercase letters, numbers, and underscores.")
    return record_id


def _record_path(storage_dir: Path | str, record_id: str) -> Path:
    safe_id = validate_record_id(record_id)
    directory = Path(storage_dir)
    path = directory / f"{safe_id}.json"
    resolved_directory = directory.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_directory:
        raise ValueError("record path escapes the requested storage directory.")
    return path


def save_json_record(
    storage_dir: Path | str,
    record_id: str,
    record: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record must be a dict.")
    path = _record_path(storage_dir, record_id)
    if path.exists() and not overwrite:
        raise ValueError(f"Record '{record_id}' already exists.")

    saved_record = deepcopy(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return saved_record


def load_json_record(storage_dir: Path | str, record_id: str) -> dict[str, Any]:
    path = _record_path(storage_dir, record_id)
    if not path.exists():
        raise FileNotFoundError(f"Record '{record_id}' does not exist.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Record '{record_id}' contains malformed JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Record '{record_id}' must contain a JSON object.")
    return raw


def list_json_records(storage_dir: Path | str) -> list[dict[str, Any]]:
    directory = Path(storage_dir)
    if not directory.exists():
        return []
    return [load_json_record(directory, path.stem) for path in sorted(directory.glob("*.json"))]


def delete_json_record(storage_dir: Path | str, record_id: str) -> bool:
    path = _record_path(storage_dir, record_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def build_audit_event(
    event_type: str,
    target_type: str,
    target_id: str,
    metadata: dict[str, Any] | None = None,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    for label, value in [
        ("event_type", event_type),
        ("target_type", target_type),
        ("target_id", target_id),
    ]:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"{label} must be a non-empty string.")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict.")

    timestamp = created_at or _now_iso()
    event_id = make_record_id(f"{timestamp}_{event_type}_{target_type}_{target_id}")
    event = {
        "event_id": event_id,
        "created_at": timestamp,
        "event_type": event_type,
        "target_type": target_type,
        "target_id": target_id,
        "metadata": deepcopy(metadata or {}),
    }
    validate_audit_event(event)
    return event


def validate_audit_event(event: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("audit event must be a dict.")
    actual_keys = set(event.keys())
    expected_keys = set(AUDIT_EVENT_KEYS)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"audit event missing required key(s): {missing}")
    if unknown:
        raise ValueError(f"audit event has unknown key(s): {unknown}")

    validated = {
        "event_id": validate_record_id(event["event_id"]),
        "created_at": _validate_timestamp(event["created_at"], "created_at"),
        "event_type": event["event_type"],
        "target_type": event["target_type"],
        "target_id": event["target_id"],
        "metadata": deepcopy(event["metadata"]),
    }
    for label in ["event_type", "target_type", "target_id"]:
        if not isinstance(validated[label], str) or validated[label].strip() == "":
            raise ValueError(f"{label} must be a non-empty string.")
    if not isinstance(validated["metadata"], dict):
        raise ValueError("metadata must be a dict.")
    return validated


def append_audit_event(
    event: dict[str, Any],
    *,
    audit_file: Path | str | None = None,
) -> dict[str, Any]:
    validated = validate_audit_event(deepcopy(event))
    path = Path(audit_file) if audit_file is not None else AUDIT_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(validated, sort_keys=True) + "\n")
    return validated


def read_audit_events(*, audit_file: Path | str | None = None) -> list[dict[str, Any]]:
    path = Path(audit_file) if audit_file is not None else AUDIT_LOG_FILE
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip() == "":
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Audit log line {line_number} contains malformed JSON.") from exc
            events.append(validate_audit_event(raw))
    return events


def compute_file_hash(path: Path | str) -> str:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Source file '{file_path}' does not exist.")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate


def _validate_config_version_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("config version record must be a dict.")
    actual_keys = set(record.keys())
    expected_keys = set(CONFIG_VERSION_KEYS)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"config version record missing required key(s): {missing}")
    if unknown:
        raise ValueError(f"config version record has unknown key(s): {unknown}")

    name = record["config_version_name"]
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("config_version_name cannot be blank.")
    source_paths = record["source_paths"]
    file_hashes = record["file_hashes"]
    if not isinstance(source_paths, dict) or not source_paths:
        raise ValueError("source_paths must be a non-empty dict.")
    if not isinstance(file_hashes, dict):
        raise ValueError("file_hashes must be a dict.")
    if set(source_paths.keys()) != set(file_hashes.keys()):
        raise ValueError("source_paths and file_hashes must contain the same source keys.")

    validated_paths: dict[str, str] = {}
    validated_hashes: dict[str, str] = {}
    for source_key in sorted(source_paths.keys()):
        if not isinstance(source_key, str) or source_key.strip() == "":
            raise ValueError("source_paths contains a blank or non-string source key.")
        source_path = source_paths[source_key]
        source_hash = file_hashes[source_key]
        if not isinstance(source_path, str) or source_path.strip() == "":
            raise ValueError(f"source_paths.{source_key} must be a non-empty string.")
        if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
            raise ValueError(f"file_hashes.{source_key} must be a SHA-256 hex digest.")
        validated_paths[source_key] = source_path
        validated_hashes[source_key] = source_hash

    notes = record["notes"]
    if not isinstance(notes, str):
        raise ValueError("notes must be a string.")

    return {
        "config_version_id": validate_record_id(record["config_version_id"]),
        "config_version_name": name,
        "created_at": _validate_timestamp(record["created_at"], "created_at"),
        "source_paths": validated_paths,
        "file_hashes": validated_hashes,
        "notes": notes,
    }


def build_config_version_record(
    *,
    config_version_name: str,
    source_paths: dict[str, str] | None = None,
    config_version_id: str | None = None,
    notes: str = "metadata only; CSVs remain authoritative",
    created_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(config_version_name, str) or config_version_name.strip() == "":
        raise ValueError("config_version_name cannot be blank.")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string.")
    resolved_paths = deepcopy(source_paths or DEFAULT_CONFIG_SOURCE_PATHS)
    if not isinstance(resolved_paths, dict) or not resolved_paths:
        raise ValueError("source_paths must be a non-empty dict.")

    path_strings: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for source_key in sorted(resolved_paths.keys()):
        source_path = resolved_paths[source_key]
        if not isinstance(source_key, str) or source_key.strip() == "":
            raise ValueError("source_paths contains a blank or non-string source key.")
        if not isinstance(source_path, str) or source_path.strip() == "":
            raise ValueError(f"source_paths.{source_key} must be a non-empty string.")
        path_strings[source_key] = source_path
        hashes[source_key] = compute_file_hash(_resolve_source_path(source_path))

    record = {
        "config_version_id": validate_record_id(config_version_id)
        if config_version_id
        else make_record_id(config_version_name),
        "config_version_name": config_version_name,
        "created_at": created_at or _now_iso(),
        "source_paths": path_strings,
        "file_hashes": hashes,
        "notes": notes,
    }
    return _validate_config_version_record(record)


def save_config_version(
    *,
    config_version_name: str,
    source_paths: dict[str, str] | None = None,
    config_version_id: str | None = None,
    notes: str = "metadata only; CSVs remain authoritative",
    overwrite: bool = False,
    storage_dir: Path | str | None = None,
    audit_file: Path | str | None = None,
) -> dict[str, Any]:
    record = build_config_version_record(
        config_version_name=config_version_name,
        source_paths=source_paths,
        config_version_id=config_version_id,
        notes=notes,
    )
    directory = Path(storage_dir) if storage_dir is not None else CONFIG_VERSION_DIR
    saved = save_json_record(directory, record["config_version_id"], record, overwrite=overwrite)
    append_audit_event(
        build_audit_event(
            "config_version_saved",
            "config_version",
            saved["config_version_id"],
            {"source": "persistence.save_config_version"},
        ),
        audit_file=audit_file,
    )
    return _validate_config_version_record(saved)


def load_config_version(
    config_version_id: str,
    *,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    directory = Path(storage_dir) if storage_dir is not None else CONFIG_VERSION_DIR
    return _validate_config_version_record(load_json_record(directory, config_version_id))


def list_config_versions(*, storage_dir: Path | str | None = None) -> list[dict[str, Any]]:
    directory = Path(storage_dir) if storage_dir is not None else CONFIG_VERSION_DIR
    records = [_validate_config_version_record(record) for record in list_json_records(directory)]
    metadata = [
        {
            "config_version_id": record["config_version_id"],
            "config_version_name": record["config_version_name"],
            "created_at": record["created_at"],
            "source_count": len(record["source_paths"]),
            "notes": record["notes"],
        }
        for record in records
    ]
    return sorted(
        metadata,
        key=lambda row: (row["config_version_name"].lower(), row["config_version_id"]),
    )


def compare_config_versions(
    config_version_id_a: str,
    config_version_id_b: str,
    *,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    version_a = load_config_version(config_version_id_a, storage_dir=storage_dir)
    version_b = load_config_version(config_version_id_b, storage_dir=storage_dir)
    keys_a = set(version_a["file_hashes"].keys())
    keys_b = set(version_b["file_hashes"].keys())

    source_comparisons: list[dict[str, Any]] = []
    for source_key in sorted(keys_a | keys_b):
        in_a = source_key in keys_a
        in_b = source_key in keys_b
        hash_a = version_a["file_hashes"].get(source_key)
        hash_b = version_b["file_hashes"].get(source_key)
        source_comparisons.append(
            {
                "source_key": source_key,
                "from_hash": hash_a,
                "to_hash": hash_b,
                "change_type": "unchanged"
                if in_a and in_b and hash_a == hash_b
                else "changed"
                if in_a and in_b
                else "missing_from_from_version"
                if not in_a
                else "missing_from_to_version",
            }
        )

    return {
        "from_config_version_id": version_a["config_version_id"],
        "to_config_version_id": version_b["config_version_id"],
        "source_comparisons": source_comparisons,
        "missing_source_keys": {
            "from_version": sorted(keys_b - keys_a),
            "to_version": sorted(keys_a - keys_b),
        },
    }
