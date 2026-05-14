from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from historical_tracking import (
    build_historical_snapshot,
    compare_historical_snapshots,
    delete_historical_snapshot,
    list_historical_snapshots,
    load_historical_snapshot,
    make_run_id,
    save_historical_snapshot,
    validate_historical_snapshot,
    validate_run_id,
)


def _score(
    intern_id: str,
    *,
    final_score: float = 80.0,
    performance_index: float = 0.8,
    output_score: float = 100.0,
    efficiency_score: float = 1.0,
    accuracy_score: float = 0.9,
    contribution_modifier: float = 1.0,
    performance_category: str = "Solid",
):
    return SimpleNamespace(
        summary={
            "intern_id": intern_id,
            "final_score": final_score,
            "performance_index": performance_index,
            "output_score": output_score,
            "efficiency_score": efficiency_score,
            "accuracy_score": accuracy_score,
            "contribution_modifier": contribution_modifier,
            "performance_category": performance_category,
        }
    )


def _bundle(*scores) -> dict:
    return {"scores": {score.summary["intern_id"]: score for score in scores}}


def _snapshot(
    run_name: str = "Busy Season Week 1",
    *,
    run_id: str | None = None,
    scores: tuple | None = None,
    source_type: str = "baseline",
    scenario_id: str | None = None,
) -> dict:
    return build_historical_snapshot(
        run_name=run_name,
        run_id=run_id,
        pipeline_result_bundle=_bundle(*(scores or (_score("INT001"),))),
        source_type=source_type,
        scenario_id=scenario_id,
        created_at="2026-05-13T00:00:00+00:00",
    )


def test_run_id_generation_is_deterministic() -> None:
    assert make_run_id("Busy Season Week 1") == "busy_season_week_1"
    assert make_run_id("Busy Season Week 1") == "busy_season_week_1"


def test_blank_run_name_fails() -> None:
    with pytest.raises(ValueError, match="run_name cannot be blank"):
        make_run_id("  ")


def test_unsafe_or_empty_run_id_fails() -> None:
    with pytest.raises(ValueError, match="run_id cannot be blank"):
        validate_run_id("")
    with pytest.raises(ValueError, match="lowercase letters"):
        validate_run_id("../Bad")


def test_build_historical_snapshot_extracts_required_intern_metrics() -> None:
    snapshot = _snapshot(scores=(_score("INT001", final_score=82.4),))
    row = snapshot["intern_metrics"][0]
    assert row == {
        "intern_id": "INT001",
        "final_score": 82.4,
        "performance_index": 0.8,
        "output_score": 100.0,
        "efficiency_score": 1.0,
        "accuracy_score": 0.9,
        "contribution_modifier": 1.0,
        "performance_category": "Solid",
    }


def test_snapshot_validation_catches_missing_required_keys() -> None:
    snapshot = _snapshot()
    snapshot.pop("intern_metrics")
    with pytest.raises(ValueError, match="missing required key"):
        validate_historical_snapshot(snapshot)


def test_snapshot_validation_catches_duplicate_intern_ids() -> None:
    snapshot = _snapshot()
    snapshot["intern_metrics"].append(copy.deepcopy(snapshot["intern_metrics"][0]))
    with pytest.raises(ValueError, match="duplicate intern_id"):
        validate_historical_snapshot(snapshot)


def test_snapshot_validation_catches_nonnumeric_metric_values() -> None:
    snapshot = _snapshot()
    snapshot["intern_metrics"][0]["final_score"] = "82.4"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_historical_snapshot(snapshot)


def test_baseline_source_rejects_scenario_id() -> None:
    with pytest.raises(ValueError, match="baseline source_type requires scenario_id"):
        _snapshot(source_type="baseline", scenario_id="scenario_a")


def test_scenario_source_requires_scenario_id() -> None:
    with pytest.raises(ValueError, match="scenario source_type requires"):
        _snapshot(source_type="scenario", scenario_id=None)


def test_save_historical_snapshot_writes_json(tmp_path: Path) -> None:
    snapshot = _snapshot()
    saved = save_historical_snapshot(snapshot, storage_dir=tmp_path)
    path = tmp_path / f"{saved['run_id']}.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == saved


def test_duplicate_save_without_overwrite_fails(tmp_path: Path) -> None:
    snapshot = _snapshot()
    save_historical_snapshot(snapshot, storage_dir=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        save_historical_snapshot(snapshot, storage_dir=tmp_path)


def test_duplicate_save_with_overwrite_succeeds(tmp_path: Path) -> None:
    first = _snapshot(scores=(_score("INT001", final_score=80.0),))
    second = _snapshot(scores=(_score("INT001", final_score=90.0),))
    save_historical_snapshot(first, storage_dir=tmp_path)
    saved = save_historical_snapshot(second, overwrite=True, storage_dir=tmp_path)
    assert saved["intern_metrics"][0]["final_score"] == 90.0


def test_load_historical_snapshot_returns_saved_snapshot(tmp_path: Path) -> None:
    snapshot = _snapshot()
    saved = save_historical_snapshot(snapshot, storage_dir=tmp_path)
    assert load_historical_snapshot(saved["run_id"], storage_dir=tmp_path) == saved


def test_loading_missing_snapshot_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_historical_snapshot("missing_run", storage_dir=tmp_path)


def test_malformed_json_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{bad-json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_historical_snapshot("bad", storage_dir=tmp_path)


def test_list_historical_snapshots_returns_stable_sorted_metadata(tmp_path: Path) -> None:
    beta = _snapshot("Beta Run")
    alpha = _snapshot("Alpha Run")
    save_historical_snapshot(beta, storage_dir=tmp_path)
    save_historical_snapshot(alpha, storage_dir=tmp_path)
    listed = list_historical_snapshots(storage_dir=tmp_path)
    assert [row["run_id"] for row in listed] == ["alpha_run", "beta_run"]
    assert set(listed[0]) == {
        "run_id",
        "run_name",
        "created_at",
        "source_type",
        "scenario_id",
        "intern_count",
    }


def test_delete_historical_snapshot_removes_saved_file(tmp_path: Path) -> None:
    saved = save_historical_snapshot(_snapshot(), storage_dir=tmp_path)
    assert delete_historical_snapshot(saved["run_id"], storage_dir=tmp_path) is True
    assert not (tmp_path / f"{saved['run_id']}.json").exists()
    assert delete_historical_snapshot(saved["run_id"], storage_dir=tmp_path) is False


def _save_compare_pair(tmp_path: Path, from_score: float, to_score: float) -> dict:
    save_historical_snapshot(
        _snapshot(
            "Week 1",
            scores=(_score("INT001", final_score=from_score),),
        ),
        storage_dir=tmp_path,
    )
    save_historical_snapshot(
        _snapshot(
            "Week 2",
            scores=(_score("INT001", final_score=to_score),),
        ),
        storage_dir=tmp_path,
    )
    return compare_historical_snapshots("week_1", "week_2", storage_dir=tmp_path)


def _final_score_trend(comparison: dict) -> dict:
    return next(
        row
        for row in comparison["metric_trends"]
        if row["intern_id"] == "INT001" and row["metric_name"] == "final_score"
    )


def test_compare_historical_snapshots_detects_metric_increase(tmp_path: Path) -> None:
    row = _final_score_trend(_save_compare_pair(tmp_path, 80.0, 90.0))
    assert row["absolute_delta"] == 10.0
    assert row["direction"] == "increase"


def test_compare_historical_snapshots_detects_metric_decrease(tmp_path: Path) -> None:
    row = _final_score_trend(_save_compare_pair(tmp_path, 90.0, 80.0))
    assert row["absolute_delta"] == -10.0
    assert row["direction"] == "decrease"


def test_compare_historical_snapshots_detects_no_change(tmp_path: Path) -> None:
    row = _final_score_trend(_save_compare_pair(tmp_path, 80.0, 80.0))
    assert row["absolute_delta"] == 0.0
    assert row["direction"] == "no_change"


def test_compare_historical_snapshots_detects_category_transition(tmp_path: Path) -> None:
    save_historical_snapshot(
        _snapshot(
            "Week 1",
            scores=(_score("INT001", performance_category="Risk"),),
        ),
        storage_dir=tmp_path,
    )
    save_historical_snapshot(
        _snapshot(
            "Week 2",
            scores=(_score("INT001", performance_category="Top"),),
        ),
        storage_dir=tmp_path,
    )
    comparison = compare_historical_snapshots("week_1", "week_2", storage_dir=tmp_path)
    assert comparison["category_transitions"] == [
        {
            "intern_id": "INT001",
            "from_category": "Risk",
            "to_category": "Top",
            "changed": True,
        }
    ]


def test_compare_historical_snapshots_handles_new_intern_in_later_snapshot(tmp_path: Path) -> None:
    save_historical_snapshot(_snapshot("Week 1", scores=(_score("INT001"),)), storage_dir=tmp_path)
    save_historical_snapshot(
        _snapshot("Week 2", scores=(_score("INT001"), _score("INT002"))),
        storage_dir=tmp_path,
    )
    comparison = compare_historical_snapshots("week_1", "week_2", storage_dir=tmp_path)
    row = next(row for row in comparison["trend_classifications"] if row["intern_id"] == "INT002")
    assert row["trend_label"] == "new_in_period"


def test_compare_historical_snapshots_handles_missing_intern_in_later_snapshot(tmp_path: Path) -> None:
    save_historical_snapshot(
        _snapshot("Week 1", scores=(_score("INT001"), _score("INT002"))),
        storage_dir=tmp_path,
    )
    save_historical_snapshot(_snapshot("Week 2", scores=(_score("INT001"),)), storage_dir=tmp_path)
    comparison = compare_historical_snapshots("week_1", "week_2", storage_dir=tmp_path)
    row = next(row for row in comparison["trend_classifications"] if row["intern_id"] == "INT002")
    assert row["trend_label"] == "missing_in_period"


def test_historical_functions_do_not_mutate_input_bundles() -> None:
    bundle = _bundle(_score("INT001"))
    before = copy.deepcopy(bundle)
    build_historical_snapshot(run_name="No Mutation", pipeline_result_bundle=bundle)
    assert bundle == before


def test_historical_save_does_not_write_to_data_config_or_scenarios(tmp_path: Path) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_before = sorted(path.name for path in (base_dir / "data").iterdir())
    config_before = sorted(path.name for path in (base_dir / "config").iterdir())
    scenarios_dir = base_dir / "scenarios"
    scenarios_before = (
        sorted(path.name for path in scenarios_dir.iterdir())
        if scenarios_dir.exists()
        else []
    )

    save_historical_snapshot(_snapshot(), storage_dir=tmp_path)

    assert sorted(path.name for path in (base_dir / "data").iterdir()) == data_before
    assert sorted(path.name for path in (base_dir / "config").iterdir()) == config_before
    scenarios_after = (
        sorted(path.name for path in scenarios_dir.iterdir())
        if scenarios_dir.exists()
        else []
    )
    assert scenarios_after == scenarios_before
