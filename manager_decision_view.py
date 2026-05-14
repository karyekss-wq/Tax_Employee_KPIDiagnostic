from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from cross_intern_patterns import build_cross_intern_patterns
from delta_analysis import build_simulation_deltas
from historical_tracking import (
    build_historical_snapshot,
    compare_historical_snapshots,
    delete_historical_snapshot,
    list_historical_snapshots,
    save_historical_snapshot,
)
from manager_actions import build_manager_actions
from persistence import (
    compare_config_versions,
    ensure_storage_dirs,
    list_config_versions,
    read_audit_events,
    save_config_version,
)
from scenario_state import (
    delete_scenario,
    list_scenarios,
    load_scenario,
    run_saved_scenario,
    save_scenario,
)
from scoring import ScoreResults
from simulation import run_simulation
from summary_layer import (
    summarize_historical_comparison,
    summarize_manager_actions,
    summarize_simulation_deltas,
    summarize_system_patterns,
)


def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a dict.")
    if key not in mapping:
        raise ValueError(f"{label} is missing required key '{key}'.")
    return mapping[key]


def build_decision_summary(
    *,
    results_by_intern: dict[str, ScoreResults],
    patterns_payload: dict[str, Any],
    actions_payload: dict[str, Any],
    saved_scenarios: list[dict[str, Any]],
    historical_snapshots: list[dict[str, Any]],
) -> dict[str, int]:
    pattern_summary = _require_key(patterns_payload, "pattern_summary", "patterns_payload")
    action_summary = _require_key(actions_payload, "action_summary", "actions_payload")
    return {
        "intern_count": len(results_by_intern),
        "highest_priority_action_count": int(
            _require_key(action_summary, "high_priority_count", "action_summary")
        ),
        "systemic_pattern_count": int(
            _require_key(pattern_summary, "systemic_count", "pattern_summary")
        ),
        "saved_scenario_count": len(saved_scenarios),
        "historical_snapshot_count": len(historical_snapshots),
    }


def group_changes_by_type(changes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in changes:
        if not isinstance(row, dict):
            raise ValueError("change rows must be dicts.")
        change_type = _require_key(row, "change_type", "change row")
        grouped.setdefault(str(change_type), []).append(dict(row))
    return {key: grouped[key] for key in sorted(grouped.keys())}


def build_baseline_vs_simulated_rows(simulation_result: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = _require_key(simulation_result, "baseline", "simulation_result")
    simulated = _require_key(simulation_result, "simulated", "simulation_result")
    baseline_scores = _require_key(baseline, "scores", "simulation_result.baseline")
    simulated_scores = _require_key(simulated, "scores", "simulation_result.simulated")

    baseline_ids = set(baseline_scores.keys())
    simulated_ids = set(simulated_scores.keys())
    if baseline_ids != simulated_ids:
        raise ValueError(
            "Baseline and simulated score intern IDs differ. "
            f"baseline_only={sorted(baseline_ids - simulated_ids)}, "
            f"simulated_only={sorted(simulated_ids - baseline_ids)}"
        )

    rows: list[dict[str, Any]] = []
    for intern_id in sorted(baseline_ids):
        baseline_summary = getattr(baseline_scores[intern_id], "summary", None)
        simulated_summary = getattr(simulated_scores[intern_id], "summary", None)
        if not isinstance(baseline_summary, dict) or not isinstance(simulated_summary, dict):
            raise ValueError(f"Missing summary for intern {intern_id}.")
        rows.append(
            {
                "Intern ID": str(intern_id),
                "Baseline Final Score": float(
                    _require_key(baseline_summary, "final_score", "baseline summary")
                ),
                "Simulated Final Score": float(
                    _require_key(simulated_summary, "final_score", "simulated summary")
                ),
                "Baseline Category": _require_key(
                    baseline_summary, "performance_category", "baseline summary"
                ),
                "Simulated Category": _require_key(
                    simulated_summary, "performance_category", "simulated summary"
                ),
            }
        )
    return rows


def build_delta_preview(deltas: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    metric_deltas = _require_key(deltas, "metric_deltas", "deltas")
    category_changes = _require_key(deltas, "category_changes", "deltas")
    action_changes = _require_key(deltas, "action_changes", "deltas")
    pattern_changes = _require_key(deltas, "pattern_changes", "deltas")

    return {
        "changed_final_scores": [
            dict(row)
            for row in metric_deltas
            if row["metric_name"] == "final_score" and row["direction"] != "no_change"
        ],
        "category_transitions": [dict(row) for row in category_changes if row["changed"]],
        "action_changes": [
            dict(row)
            for row in action_changes
            if row["change_type"] in {"added", "removed", "priority_changed"}
        ],
        "pattern_changes": [
            dict(row)
            for row in pattern_changes
            if row["change_type"]
            in {"introduced", "resolved", "scope_changed", "frequency_changed"}
        ],
    }


def extract_historical_trend_summary(comparison: dict[str, Any]) -> dict[str, int]:
    summary = _require_key(comparison, "trend_summary", "historical comparison")
    required = [
        "improving_count",
        "deteriorating_count",
        "unchanged_count",
        "new_in_period_count",
        "missing_in_period_count",
    ]
    return {key: int(_require_key(summary, key, "trend_summary")) for key in required}


def build_persistence_status(
    *,
    storage_paths: dict[str, Any],
    saved_scenarios: list[dict[str, Any]],
    historical_snapshots: list[dict[str, Any]],
    config_versions: list[dict[str, Any]],
    audit_events: list[dict[str, Any]],
) -> dict[str, Any]:
    required_paths = ["scenarios", "history", "audit", "config_versions"]
    path_summary = {
        key: str(_require_key(storage_paths, key, "storage_paths")) for key in required_paths
    }
    return {
        "storage_paths": path_summary,
        "scenario_count": len(saved_scenarios),
        "history_snapshot_count": len(historical_snapshots),
        "config_version_count": len(config_versions),
        "audit_event_count": len(audit_events),
    }


def format_recent_audit_events(
    audit_events: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("limit must be nonnegative.")
    rows: list[dict[str, Any]] = []
    for event in audit_events[-limit:]:
        rows.append(
            {
                "Created At": _require_key(event, "created_at", "audit event"),
                "Event Type": _require_key(event, "event_type", "audit event"),
                "Target Type": _require_key(event, "target_type", "audit event"),
                "Target ID": _require_key(event, "target_id", "audit event"),
            }
        )
    return rows


def build_config_version_comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    source_comparisons = _require_key(
        comparison, "source_comparisons", "config version comparison"
    )
    if not isinstance(source_comparisons, list):
        raise ValueError("config version comparison source_comparisons must be a list.")
    rows: list[dict[str, Any]] = []
    for row in source_comparisons:
        if not isinstance(row, dict):
            raise ValueError("config version comparison rows must be dicts.")
        rows.append(
            {
                "Source": _require_key(row, "source_key", "source comparison"),
                "Change Type": _require_key(row, "change_type", "source comparison"),
                "From Hash": _require_key(row, "from_hash", "source comparison"),
                "To Hash": _require_key(row, "to_hash", "source comparison"),
            }
        )
    return rows


def build_override_dicts(
    class_config: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    class_editor: pd.DataFrame,
    adjustment_editor: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    base_class = class_config.set_index("task_class")
    edited_class = class_editor.set_index("task_class")
    expected_hours_overrides: dict[str, Any] = {}
    class_weight_overrides: dict[str, Any] = {}
    for task_class in edited_class.index:
        if edited_class.loc[task_class, "base_expected_hours"] != base_class.loc[task_class, "base_expected_hours"]:
            expected_hours_overrides[str(task_class)] = edited_class.loc[
                task_class, "base_expected_hours"
            ]
        if edited_class.loc[task_class, "base_class_weight"] != base_class.loc[task_class, "base_class_weight"]:
            class_weight_overrides[str(task_class)] = edited_class.loc[
                task_class, "base_class_weight"
            ]

    base_adjustment = adjustment_config.set_index("adjustment_code")
    edited_adjustment = adjustment_editor.set_index("adjustment_code")
    adjustment_overrides: dict[str, Any] = {}
    for adjustment_code in edited_adjustment.index:
        if edited_adjustment.loc[adjustment_code, "multiplier_add"] != base_adjustment.loc[adjustment_code, "multiplier_add"]:
            adjustment_overrides[str(adjustment_code)] = edited_adjustment.loc[
                adjustment_code, "multiplier_add"
            ]

    return {
        "class_expected_hours_overrides": expected_hours_overrides,
        "class_weight_overrides": class_weight_overrides,
        "adjustment_multiplier_overrides": adjustment_overrides,
    }


def apply_loaded_overrides(
    class_config: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    overrides: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_editor_source = class_config[
        ["task_class", "class_name", "base_expected_hours", "base_class_weight"]
    ].copy()
    adjustment_editor_source = adjustment_config[
        ["adjustment_code", "label", "multiplier_add", "min_bound", "max_bound"]
    ].copy()
    if not overrides:
        return class_editor_source, adjustment_editor_source

    for task_class, value in overrides["class_expected_hours_overrides"].items():
        class_editor_source.loc[
            class_editor_source["task_class"] == task_class, "base_expected_hours"
        ] = value
    for task_class, value in overrides["class_weight_overrides"].items():
        class_editor_source.loc[
            class_editor_source["task_class"] == task_class, "base_class_weight"
        ] = value
    for adjustment_code, value in overrides["adjustment_multiplier_overrides"].items():
        adjustment_editor_source.loc[
            adjustment_editor_source["adjustment_code"] == adjustment_code,
            "multiplier_add",
        ] = value
    return class_editor_source, adjustment_editor_source


def render_summary_objects(title: str, summaries: list[dict[str, Any]]) -> None:
    with st.expander(title, expanded=True):
        if not summaries:
            st.info("No generated summaries available.")
            return
        for summary in summaries:
            st.write(f"- {summary['summary_text']}")
        st.caption("Generated by deterministic templates from structured outputs.")


def render_baseline_summary(
    *,
    results_by_intern: dict[str, ScoreResults],
    patterns_payload: dict[str, Any],
    actions_payload: dict[str, Any],
    saved_scenarios: list[dict[str, Any]],
    historical_snapshots: list[dict[str, Any]],
) -> None:
    summary = build_decision_summary(
        results_by_intern=results_by_intern,
        patterns_payload=patterns_payload,
        actions_payload=actions_payload,
        saved_scenarios=saved_scenarios,
        historical_snapshots=historical_snapshots,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Interns", summary["intern_count"])
    c2.metric("High-Priority Actions", summary["highest_priority_action_count"])
    c3.metric("Systemic Patterns", summary["systemic_pattern_count"])
    c4.metric("Saved Scenarios", summary["saved_scenario_count"])
    c5.metric("History Snapshots", summary["historical_snapshot_count"])

    render_summary_objects(
        "Generated Manager Summary",
        summarize_manager_actions(actions_payload) + summarize_system_patterns(patterns_payload),
    )

    all_actions = actions_payload["intern_actions"] + actions_payload["team_actions"]
    if all_actions:
        action_rows = [
            {
                "Priority": action["priority_level"],
                "Scope": action["target_scope"],
                "Type": action["action_type"],
                "Target": action["target_id"],
                "Message": action["message"],
            }
            for action in all_actions
        ]
        st.write("Current Manager Actions")
        st.dataframe(pd.DataFrame(action_rows), hide_index=True, use_container_width=True)

    pattern_rows = [
        {
            "Pattern Type": pattern["pattern_type"],
            "Scope": pattern["scope_classification"],
            "Severity": pattern["severity"],
            "Frequency": round(float(pattern["frequency"]), 4),
            "Message": pattern["message"],
        }
        for pattern in patterns_payload["system_patterns"]
    ]
    if pattern_rows:
        st.write("Current System Patterns")
        st.dataframe(pd.DataFrame(pattern_rows), hide_index=True, use_container_width=True)


def render_scenario_library() -> list[dict[str, Any]]:
    saved_scenarios = list_scenarios()
    st.write("Saved Scenarios")
    if not saved_scenarios:
        st.info("No saved scenarios yet.")
        return saved_scenarios

    scenario_options = [row["scenario_id"] for row in saved_scenarios]
    selected_scenario_id = st.selectbox(
        "Saved scenario",
        options=scenario_options,
        format_func=lambda sid: next(
            row["scenario_name"] for row in saved_scenarios if row["scenario_id"] == sid
        ),
        key="decision_saved_scenario_selector",
    )

    s1, s2, s3 = st.columns(3)
    if s1.button("Load Scenario", key="decision_load_saved_scenario"):
        try:
            selected_record = load_scenario(selected_scenario_id)
            st.session_state["loaded_scenario_id"] = selected_record["scenario_id"]
            st.session_state["loaded_scenario_name"] = selected_record["scenario_name"]
            st.session_state["loaded_scenario_overrides"] = selected_record["overrides"]
            st.rerun()
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))

    if s2.button("Run Saved Scenario", key="decision_run_saved_scenario"):
        try:
            saved_run = run_saved_scenario(selected_scenario_id)
            st.session_state["last_simulation_result"] = saved_run["simulation_result"]
            st.session_state["last_delta_result"] = saved_run["deltas"]
            st.success(f"Ran scenario {selected_scenario_id}.")
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))

    if s3.button("Delete Scenario", key="decision_delete_saved_scenario"):
        if delete_scenario(selected_scenario_id):
            st.session_state.pop("loaded_scenario_id", None)
            st.session_state.pop("loaded_scenario_name", None)
            st.session_state.pop("loaded_scenario_overrides", None)
            st.success(f"Deleted scenario {selected_scenario_id}.")
            st.rerun()
        st.warning(f"Scenario {selected_scenario_id} was not found.")

    st.dataframe(pd.DataFrame(saved_scenarios), hide_index=True, use_container_width=True)
    return saved_scenarios


def render_scenario_simulation_panel(
    *,
    class_config: pd.DataFrame,
    adjustment_config: pd.DataFrame,
) -> None:
    loaded_overrides = st.session_state.get("loaded_scenario_overrides")
    loaded_name = st.session_state.get("loaded_scenario_name", "Manager what-if scenario")
    class_editor_source, adjustment_editor_source = apply_loaded_overrides(
        class_config, adjustment_config, loaded_overrides
    )
    editor_key_suffix = st.session_state.get("loaded_scenario_id", "draft")

    with st.form("decision_scenario_simulation_form"):
        scenario_name = st.text_input("Scenario name", value=loaded_name)
        st.write("Class Expected Hours and Weights")
        class_editor = st.data_editor(
            class_editor_source,
            hide_index=True,
            use_container_width=True,
            disabled=["task_class", "class_name"],
            key=f"decision_class_editor_{editor_key_suffix}",
        )
        st.write("Adjustment Multipliers")
        adjustment_editor = st.data_editor(
            adjustment_editor_source,
            hide_index=True,
            use_container_width=True,
            disabled=["adjustment_code", "label", "min_bound", "max_bound"],
            key=f"decision_adjustment_editor_{editor_key_suffix}",
        )
        save_after_run = st.checkbox("Save scenario after run", value=False)
        overwrite_existing = st.checkbox("Overwrite existing scenario", value=False)
        submitted = st.form_submit_button("Run Simulation")

    if st.button("Reset Controls to Baseline", key="decision_reset_scenario_controls"):
        st.session_state.pop("loaded_scenario_id", None)
        st.session_state.pop("loaded_scenario_name", None)
        st.session_state.pop("loaded_scenario_overrides", None)
        st.rerun()

    if not submitted:
        return

    overrides = build_override_dicts(
        class_config, adjustment_config, class_editor, adjustment_editor
    )
    try:
        if save_after_run:
            saved_record = save_scenario(
                scenario_name=scenario_name,
                class_expected_hours_overrides=overrides["class_expected_hours_overrides"],
                class_weight_overrides=overrides["class_weight_overrides"],
                adjustment_multiplier_overrides=overrides["adjustment_multiplier_overrides"],
                overwrite=overwrite_existing,
            )
            st.success(f"Saved scenario {saved_record['scenario_id']}.")

        simulation_result = run_simulation(
            scenario_name=scenario_name,
            class_expected_hours_overrides=overrides["class_expected_hours_overrides"],
            class_weight_overrides=overrides["class_weight_overrides"],
            adjustment_multiplier_overrides=overrides["adjustment_multiplier_overrides"],
        )
        deltas = build_simulation_deltas(
            simulation_result["baseline"], simulation_result["simulated"]
        )
        st.session_state["last_simulation_result"] = simulation_result
        st.session_state["last_delta_result"] = deltas
    except ValueError as exc:
        st.error(str(exc))


def render_baseline_vs_simulated_comparison() -> None:
    simulation_result = st.session_state.get("last_simulation_result")
    deltas = st.session_state.get("last_delta_result")
    if not simulation_result or not deltas:
        st.info("Run or load a scenario to see baseline vs simulated comparison.")
        return

    comparison_rows = build_baseline_vs_simulated_rows(simulation_result)
    st.write("Baseline vs Simulated Intern Comparison")
    st.dataframe(
        pd.DataFrame(comparison_rows).round(4),
        hide_index=True,
        use_container_width=True,
    )

    preview = build_delta_preview(deltas)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Changed Scores", len(preview["changed_final_scores"]))
    m2.metric("Category Transitions", len(preview["category_transitions"]))
    m3.metric("Action Changes", len(preview["action_changes"]))
    m4.metric("Pattern Changes", len(preview["pattern_changes"]))


def render_delta_impact_preview() -> None:
    deltas = st.session_state.get("last_delta_result")
    if not deltas:
        st.info("Run or load a scenario to see delta impact.")
        return

    preview = build_delta_preview(deltas)
    render_summary_objects(
        "Generated Delta Summary",
        summarize_simulation_deltas(deltas),
    )
    sections = [
        ("Changed Final Scores", preview["changed_final_scores"]),
        ("Category Transitions", preview["category_transitions"]),
        ("Metric Deltas", deltas["metric_deltas"]),
        ("Driver Changes", [row for row in deltas["driver_changes"] if row["changed"]]),
    ]
    for label, rows in sections:
        st.write(label)
        if rows:
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
        else:
            st.info(f"No {label.lower()}.")


def render_action_impact_preview() -> None:
    deltas = st.session_state.get("last_delta_result")
    if not deltas:
        st.info("Run or load a scenario to see action impact.")
        return

    grouped = group_changes_by_type(deltas["action_changes"])
    for change_type in ["added", "removed", "priority_changed", "unchanged"]:
        rows = grouped.get(change_type, [])
        with st.expander(f"{change_type} actions ({len(rows)})", expanded=change_type != "unchanged"):
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info(f"No {change_type} actions.")


def render_pattern_shift_preview() -> None:
    deltas = st.session_state.get("last_delta_result")
    if not deltas:
        st.info("Run or load a scenario to see pattern shifts.")
        return

    grouped = group_changes_by_type(deltas["pattern_changes"])
    for change_type in ["introduced", "resolved", "scope_changed", "frequency_changed", "unchanged"]:
        rows = grouped.get(change_type, [])
        with st.expander(f"{change_type} patterns ({len(rows)})", expanded=change_type != "unchanged"):
            if rows:
                st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
            else:
                st.info(f"No {change_type} patterns.")


def render_historical_tracking_panel(results_by_intern: dict[str, ScoreResults]) -> None:
    with st.form("decision_historical_snapshot_form"):
        run_name = st.text_input("Historical run name", value="Current Baseline Run")
        overwrite_run = st.checkbox("Overwrite existing run", value=False)
        save_run = st.form_submit_button("Save Current Baseline Snapshot")

    if save_run:
        try:
            snapshot = build_historical_snapshot(
                run_name=run_name,
                pipeline_result_bundle={"scores": results_by_intern},
                source_type="baseline",
            )
            saved = save_historical_snapshot(snapshot, overwrite=overwrite_run)
            st.success(f"Saved historical snapshot {saved['run_id']}.")
        except ValueError as exc:
            st.error(str(exc))

    snapshots = list_historical_snapshots()
    if not snapshots:
        st.info("No historical snapshots saved yet.")
        return

    st.dataframe(pd.DataFrame(snapshots), hide_index=True, use_container_width=True)
    snapshot_ids = [row["run_id"] for row in snapshots]
    c1, c2, c3 = st.columns([2, 2, 1])
    from_run = c1.selectbox("From snapshot", options=snapshot_ids, key="decision_history_from_run")
    to_run = c2.selectbox("To snapshot", options=snapshot_ids, key="decision_history_to_run")
    selected_for_delete = c3.selectbox(
        "Delete snapshot", options=snapshot_ids, key="decision_history_delete_run"
    )

    b1, b2 = st.columns(2)
    if b1.button("Compare Snapshots", key="decision_compare_history_snapshots"):
        try:
            comparison = compare_historical_snapshots(from_run, to_run)
            st.session_state["last_history_comparison"] = comparison
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))

    if b2.button("Delete Selected Snapshot", key="decision_delete_history_snapshot"):
        if delete_historical_snapshot(selected_for_delete):
            st.session_state.pop("last_history_comparison", None)
            st.success(f"Deleted historical snapshot {selected_for_delete}.")
            st.rerun()
        st.warning(f"Historical snapshot {selected_for_delete} was not found.")

    comparison = st.session_state.get("last_history_comparison")
    if not comparison:
        return

    summary = extract_historical_trend_summary(comparison)
    render_summary_objects(
        "Generated Historical Summary",
        summarize_historical_comparison(comparison),
    )
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Improving", summary["improving_count"])
    h2.metric("Deteriorating", summary["deteriorating_count"])
    h3.metric("Unchanged", summary["unchanged_count"])
    h4.metric("New", summary["new_in_period_count"])
    h5.metric("Missing", summary["missing_in_period_count"])

    final_score_rows = [
        row for row in comparison["metric_trends"] if row["metric_name"] == "final_score"
    ]
    if final_score_rows:
        st.write("Final Score Trends")
        st.dataframe(
            pd.DataFrame(final_score_rows).round(4),
            hide_index=True,
            use_container_width=True,
        )

    category_rows = [row for row in comparison["category_transitions"] if row["changed"]]
    if category_rows:
        st.write("Category Transitions")
        st.dataframe(pd.DataFrame(category_rows), hide_index=True, use_container_width=True)


def render_persistence_audit_panel(
    *,
    saved_scenarios: list[dict[str, Any]],
    historical_snapshots: list[dict[str, Any]],
) -> None:
    try:
        storage_paths = ensure_storage_dirs()
        config_versions = list_config_versions()
        audit_events = read_audit_events()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    status = build_persistence_status(
        storage_paths=storage_paths,
        saved_scenarios=saved_scenarios,
        historical_snapshots=historical_snapshots,
        config_versions=config_versions,
        audit_events=audit_events,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenarios", status["scenario_count"])
    c2.metric("History Snapshots", status["history_snapshot_count"])
    c3.metric("Config Versions", status["config_version_count"])
    c4.metric("Audit Events", status["audit_event_count"])

    with st.expander("Storage Paths", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Storage Area": key, "Path": value}
                    for key, value in status["storage_paths"].items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.form("decision_config_version_form"):
        version_name = st.text_input("Config version name", value="Current Baseline Config")
        overwrite_version = st.checkbox("Overwrite existing config version", value=False)
        save_version = st.form_submit_button("Create Config Version Metadata")

    if save_version:
        try:
            saved = save_config_version(
                config_version_name=version_name,
                overwrite=overwrite_version,
            )
            st.success(f"Saved config version {saved['config_version_id']}.")
            st.rerun()
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))

    config_versions = list_config_versions()
    if config_versions:
        st.write("Config Versions")
        st.dataframe(pd.DataFrame(config_versions), hide_index=True, use_container_width=True)
    else:
        st.info("No config versions saved yet.")

    if len(config_versions) >= 2:
        version_ids = [row["config_version_id"] for row in config_versions]
        c_from, c_to = st.columns(2)
        from_version = c_from.selectbox(
            "From config version",
            options=version_ids,
            key="decision_config_version_from",
        )
        to_version = c_to.selectbox(
            "To config version",
            options=version_ids,
            key="decision_config_version_to",
        )
        if st.button("Compare Config Versions", key="decision_compare_config_versions"):
            try:
                comparison = compare_config_versions(from_version, to_version)
                st.session_state["last_config_version_comparison"] = comparison
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

    comparison = st.session_state.get("last_config_version_comparison")
    if comparison:
        st.write("Config Version Comparison")
        st.dataframe(
            pd.DataFrame(build_config_version_comparison_rows(comparison)),
            hide_index=True,
            use_container_width=True,
        )

    recent_events = format_recent_audit_events(read_audit_events(), limit=5)
    st.write("Recent Audit Events")
    if recent_events:
        st.dataframe(pd.DataFrame(recent_events), hide_index=True, use_container_width=True)
    else:
        st.info("No audit events recorded yet.")


def render_manager_decision_dashboard(
    *,
    results_by_intern: dict[str, ScoreResults],
    default_intern_id: str,
    class_config_loader: Callable[[], pd.DataFrame],
    adjustment_config_loader: Callable[[], pd.DataFrame],
) -> None:
    st.header("Manager Decision Dashboard")
    st.caption("Decision-focused view built from deterministic scoring, simulation, delta, scenario, and history outputs.")

    if not results_by_intern:
        st.info("No intern results available.")
        return

    patterns_payload = build_cross_intern_patterns(results_by_intern)
    actions_payload = build_manager_actions(results_by_intern)
    saved_scenarios = list_scenarios()
    historical_snapshots = list_historical_snapshots()

    tabs = st.tabs(
        [
            "Decision Summary",
            "Scenario Simulation",
            "Delta Impact",
            "Scenario Library",
            "Historical Tracking",
            "Persistence & Audit",
        ]
    )

    with tabs[0]:
        render_baseline_summary(
            results_by_intern=results_by_intern,
            patterns_payload=patterns_payload,
            actions_payload=actions_payload,
            saved_scenarios=saved_scenarios,
            historical_snapshots=historical_snapshots,
        )

    with tabs[1]:
        render_scenario_simulation_panel(
            class_config=class_config_loader(),
            adjustment_config=adjustment_config_loader(),
        )
        render_baseline_vs_simulated_comparison()

    with tabs[2]:
        render_delta_impact_preview()
        render_action_impact_preview()
        render_pattern_shift_preview()

    with tabs[3]:
        render_scenario_library()

    with tabs[4]:
        render_historical_tracking_panel(results_by_intern)

    with tabs[5]:
        render_persistence_audit_panel(
            saved_scenarios=saved_scenarios,
            historical_snapshots=historical_snapshots,
        )
