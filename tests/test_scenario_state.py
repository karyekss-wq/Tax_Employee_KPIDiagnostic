from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scenario_state
from scenario_state import (
    build_scenario_record,
    delete_scenario,
    get_baseline_state,
    list_scenarios,
    load_scenario,
    make_scenario_id,
    reset_to_baseline,
    run_saved_scenario,
    save_scenario,
    validate_scenario_id,
    validate_scenario_record,
)


def _overrides() -> dict:
    return {
        "class_expected_hours_overrides": {"A": 2.2},
        "class_weight_overrides": {"A": 1.15},
        "adjustment_multiplier_overrides": {"multi_state": 0.2},
    }


def _save_sample(tmp_path: Path, name: str = "Reduce Class A") -> dict:
    overrides = _overrides()
    return save_scenario(
        scenario_name=name,
        class_expected_hours_overrides=overrides["class_expected_hours_overrides"],
        class_weight_overrides=overrides["class_weight_overrides"],
        adjustment_multiplier_overrides=overrides["adjustment_multiplier_overrides"],
        storage_dir=tmp_path,
    )


def test_scenario_id_generation_is_deterministic() -> None:
    assert make_scenario_id("Reduce Class A Expected Hours") == "reduce_class_a_expected_hours"
    assert make_scenario_id("Reduce Class A Expected Hours") == "reduce_class_a_expected_hours"


def test_blank_scenario_name_fails() -> None:
    with pytest.raises(ValueError, match="scenario_name cannot be blank"):
        make_scenario_id("   ")


def test_unsafe_or_empty_scenario_id_fails() -> None:
    with pytest.raises(ValueError, match="scenario_id cannot be blank"):
        validate_scenario_id("")
    with pytest.raises(ValueError, match="lowercase letters"):
        validate_scenario_id("../Bad")


def test_scenario_record_has_required_schema() -> None:
    record = build_scenario_record(
        scenario_name="Reduce Class A",
        class_expected_hours_overrides={"A": 2.2},
        class_weight_overrides={"A": 1.15},
        adjustment_multiplier_overrides={"multi_state": 0.2},
    )
    assert set(record) == {
        "scenario_id",
        "scenario_name",
        "created_at",
        "updated_at",
        "overrides",
    }
    assert set(record["overrides"]) == {
        "class_expected_hours_overrides",
        "class_weight_overrides",
        "adjustment_multiplier_overrides",
    }


def test_save_scenario_writes_json(tmp_path: Path) -> None:
    record = _save_sample(tmp_path)
    path = tmp_path / f"{record['scenario_id']}.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_duplicate_save_without_overwrite_fails(tmp_path: Path) -> None:
    _save_sample(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        _save_sample(tmp_path)


def test_duplicate_save_with_overwrite_succeeds(tmp_path: Path) -> None:
    first = _save_sample(tmp_path)
    second = save_scenario(
        scenario_name="Reduce Class A",
        class_expected_hours_overrides={"A": 2.4},
        class_weight_overrides={"A": 1.2},
        adjustment_multiplier_overrides={"multi_state": 0.25},
        overwrite=True,
        storage_dir=tmp_path,
    )
    assert second["scenario_id"] == first["scenario_id"]
    assert second["created_at"] == first["created_at"]
    assert second["overrides"]["class_expected_hours_overrides"] == {"A": 2.4}


def test_load_scenario_returns_exact_saved_override_state(tmp_path: Path) -> None:
    saved = _save_sample(tmp_path)
    loaded = load_scenario(saved["scenario_id"], storage_dir=tmp_path)
    assert loaded["overrides"] == _overrides()


def test_list_scenarios_returns_stable_sorted_metadata(tmp_path: Path) -> None:
    beta = _save_sample(tmp_path, "Beta Scenario")
    alpha = _save_sample(tmp_path, "Alpha Scenario")
    listed = list_scenarios(storage_dir=tmp_path)
    assert [row["scenario_id"] for row in listed] == [
        alpha["scenario_id"],
        beta["scenario_id"],
    ]
    assert set(listed[0]) == {
        "scenario_id",
        "scenario_name",
        "created_at",
        "updated_at",
        "override_count",
    }


def test_delete_scenario_removes_saved_file(tmp_path: Path) -> None:
    saved = _save_sample(tmp_path)
    assert delete_scenario(saved["scenario_id"], storage_dir=tmp_path) is True
    assert not (tmp_path / f"{saved['scenario_id']}.json").exists()
    assert delete_scenario(saved["scenario_id"], storage_dir=tmp_path) is False


def test_loading_missing_scenario_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_scenario("missing_scenario", storage_dir=tmp_path)


def test_malformed_json_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_scenario("bad", storage_dir=tmp_path)


def test_unknown_override_section_fails_clearly() -> None:
    record = build_scenario_record(scenario_name="Valid")
    record["overrides"]["unknown"] = {}
    with pytest.raises(ValueError, match="Unknown override section"):
        validate_scenario_record(record)


def test_nonnumeric_override_value_fails_clearly() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        build_scenario_record(
            scenario_name="Bad Numeric",
            class_expected_hours_overrides={"A": "2.2"},
        )


def test_baseline_reset_returns_empty_override_state_and_does_not_mutate_config_data() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    config_before = {
        path.name: path.read_bytes() for path in sorted((base_dir / "config").glob("*.csv"))
    }
    data_before = {
        path.name: path.read_bytes() for path in sorted((base_dir / "data").glob("*.csv"))
    }

    baseline = reset_to_baseline()

    assert baseline == get_baseline_state()
    assert baseline["overrides"] == {
        "class_expected_hours_overrides": {},
        "class_weight_overrides": {},
        "adjustment_multiplier_overrides": {},
    }
    assert config_before == {
        path.name: path.read_bytes() for path in sorted((base_dir / "config").glob("*.csv"))
    }
    assert data_before == {
        path.name: path.read_bytes() for path in sorted((base_dir / "data").glob("*.csv"))
    }


def test_scenario_save_does_not_write_to_data_or_config(tmp_path: Path) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    config_before = sorted(path.name for path in (base_dir / "config").iterdir())
    data_before = sorted(path.name for path in (base_dir / "data").iterdir())

    _save_sample(tmp_path)

    assert sorted(path.name for path in (base_dir / "config").iterdir()) == config_before
    assert sorted(path.name for path in (base_dir / "data").iterdir()) == data_before


def test_run_saved_scenario_calls_existing_run_simulation_path(tmp_path: Path, monkeypatch) -> None:
    saved = _save_sample(tmp_path)
    calls = []

    def fake_run_simulation(**kwargs):
        calls.append(kwargs)
        return {"baseline": {"scores": {}}, "simulated": {"scores": {}}}

    monkeypatch.setattr(scenario_state, "run_simulation", fake_run_simulation)

    result = run_saved_scenario(saved["scenario_id"], storage_dir=tmp_path, include_deltas=False)

    assert result["scenario"]["scenario_id"] == saved["scenario_id"]
    assert calls == [
        {
            "scenario_name": saved["scenario_name"],
            "class_expected_hours_overrides": {"A": 2.2},
            "class_weight_overrides": {"A": 1.15},
            "adjustment_multiplier_overrides": {"multi_state": 0.2},
        }
    ]


def test_scenario_functions_do_not_mutate_input_override_dictionaries(tmp_path: Path) -> None:
    class_expected = {"A": 2.2}
    class_weights = {"A": 1.15}
    adjustments = {"multi_state": 0.2}
    before = copy.deepcopy((class_expected, class_weights, adjustments))

    save_scenario(
        scenario_name="No Mutation",
        class_expected_hours_overrides=class_expected,
        class_weight_overrides=class_weights,
        adjustment_multiplier_overrides=adjustments,
        storage_dir=tmp_path,
    )

    assert (class_expected, class_weights, adjustments) == before


def test_scenario_ids_do_not_affect_config_keys_inside_overrides(tmp_path: Path) -> None:
    saved = save_scenario(
        scenario_name="Normalize This Name",
        class_expected_hours_overrides={"Class A": 2.2},
        class_weight_overrides={"A-Exact": 1.15},
        adjustment_multiplier_overrides={"multi_state": 0.2},
        storage_dir=tmp_path,
    )
    loaded = load_scenario(saved["scenario_id"], storage_dir=tmp_path)
    assert saved["scenario_id"] == "normalize_this_name"
    assert loaded["overrides"]["class_expected_hours_overrides"] == {"Class A": 2.2}
    assert loaded["overrides"]["class_weight_overrides"] == {"A-Exact": 1.15}
