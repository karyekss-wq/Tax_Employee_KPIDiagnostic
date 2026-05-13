from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cross_intern_patterns import build_cross_intern_patterns
from diagnostic_insights import build_diagnostic_insights
from manager_actions import build_manager_actions
from scoring import ScoreResults, load_csvs, run_scoring_for_inputs


@dataclass(frozen=True)
class PipelineInputs:
    class_config: pd.DataFrame
    adjustment_config: pd.DataFrame
    tasks: pd.DataFrame
    flags: pd.DataFrame


def load_pipeline_inputs() -> PipelineInputs:
    """
    Load the canonical baseline CSV inputs used by the dashboard pipeline.
    """
    class_config, adjustment_config, tasks, flags = load_csvs()
    return PipelineInputs(
        class_config=class_config,
        adjustment_config=adjustment_config,
        tasks=tasks,
        flags=flags,
    )


def run_full_pipeline(
    *,
    class_config: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    tasks: pd.DataFrame,
    flags: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run the existing deterministic analytics stack from in-memory inputs.
    """
    scores: dict[str, ScoreResults] = run_scoring_for_inputs(
        class_config=class_config,
        adjustment_config=adjustment_config,
        tasks=tasks,
        flags=flags,
    )
    diagnostics = {
        intern_id: build_diagnostic_insights(scores, intern_id)
        for intern_id in sorted(scores.keys())
    }
    normalized_insights = [
        record
        for intern_id in sorted(diagnostics.keys())
        for record in diagnostics[intern_id]["normalized_insights"]
    ]
    patterns = build_cross_intern_patterns(scores)
    actions = build_manager_actions(scores)

    return {
        "scores": scores,
        "diagnostics": diagnostics,
        "normalized_insights": normalized_insights,
        "system_patterns": patterns,
        "manager_actions": actions,
    }
