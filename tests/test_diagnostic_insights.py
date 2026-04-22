from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass

import pandas as pd

from diagnostic_insights import (
    build_diagnostic_insights,
    get_metric_peer_context,
)
from diagnostic_validation import validate_diagnostic_insights
from scoring import run_scoring


@dataclass
class FakeResult:
    task_metrics: pd.DataFrame
    summary: dict
    attribution: dict


def _base_task_metrics(task_ids: list[str], intern_id: str) -> pd.DataFrame:
    return pd.DataFrame({"task_id": task_ids, "intern_id": [intern_id] * len(task_ids)})


def _base_attribution() -> dict:
    return {
        "output_attribution": {
            "by_class": [
                {
                    "task_class": "B",
                    "class_name": "Business",
                    "output_contribution": 3.2,
                }
            ],
            "by_adjustment": [
                {
                    "adjustment_code": "investments",
                    "output_effect": 0.8,
                }
            ],
        },
        "efficiency_attribution": {
            "largest_overruns": [
                {
                    "task_id": "T101",
                    "task_class": "B",
                    "overrun_hours": 2.0,
                }
            ],
            "largest_underruns": [
                {
                    "task_id": "T102",
                    "task_class": "A",
                    "underrun_hours": 1.1,
                }
            ],
        },
        "accuracy_attribution": {
            "top_error_drivers": [
                {
                    "task_id": "T101",
                    "task_class": "B",
                    "weighted_errors": 2.5,
                }
            ],
            "by_severity": [
                {"severity": "minor", "weighted_error_impact": 0.5},
                {"severity": "major", "weighted_error_impact": 1.5},
            ],
        },
        "contribution_attribution": {
            "positive_by_type": [
                {
                    "flag_type": "helped_peer",
                    "modifier_effect": 0.015,
                }
            ],
            "negative_by_type": [
                {
                    "flag_type": "rework_requested",
                    "modifier_effect": -0.04,
                }
            ],
            "raw_positive_effect": 0.015,
            "raw_negative_effect": -0.04,
        },
    }


def _make_fake_result(
    intern_id: str,
    *,
    final_score: float,
    performance_index: float,
    output_score: float,
    efficiency_score: float,
    accuracy_score: float,
    contribution_modifier: float,
    total_weighted_errors: float = 0.0,
    positive_flags: int = 1,
    negative_flags: int = 1,
    attribution_overrides: dict | None = None,
    task_ids: list[str] | None = None,
) -> FakeResult:
    attribution = _base_attribution()
    if attribution_overrides:
        for key, value in attribution_overrides.items():
            attribution[key] = value

    return FakeResult(
        task_metrics=_base_task_metrics(task_ids or ["T101", "T102"], intern_id),
        summary={
            "intern_id": intern_id,
            "final_score": final_score,
            "performance_index": performance_index,
            "output_score": output_score,
            "efficiency_score": efficiency_score,
            "accuracy_score": accuracy_score,
            "contribution_modifier": contribution_modifier,
            "total_weighted_errors": total_weighted_errors,
            "positive_flags": positive_flags,
            "negative_flags": negative_flags,
        },
        attribution=attribution,
    )


class DiagnosticInsightsTests(unittest.TestCase):
    def test_live_data_validation_harness_passes(self) -> None:
        results = run_scoring()
        errors = validate_diagnostic_insights(results)
        self.assertEqual([], errors, f"Expected no validation errors, found: {errors}")

    def test_live_data_builds_for_all_interns(self) -> None:
        results = run_scoring()
        for intern_id in sorted(results.keys()):
            insights = build_diagnostic_insights(results, intern_id)
            self.assertIn("intern_summary", insights)
            self.assertIn("cross_intern_positioning", insights)
            self.assertIn("attribution_explanations", insights)

    def test_metric_rank_tie_break_uses_intern_id(self) -> None:
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=5.0,
                performance_index=0.8,
                output_score=2.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=5.0,
                performance_index=0.7,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
        }
        from diagnostic_insights import build_cross_intern_comparison

        comparison_df = build_cross_intern_comparison(results)
        rank1 = get_metric_peer_context(comparison_df, "INT001", "final_score")["rank"]
        rank2 = get_metric_peer_context(comparison_df, "INT002", "final_score")["rank"]
        self.assertEqual(1, rank1)
        self.assertEqual(2, rank2)

    def test_strength_tie_break_prefers_output_score(self) -> None:
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=10.0,
                performance_index=0.9,
                output_score=3.0,
                efficiency_score=3.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.0,
                performance_index=0.6,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT003": _make_fake_result(
                "INT003",
                final_score=1.0,
                performance_index=0.6,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
        }
        insights = build_diagnostic_insights(results, "INT001")
        strength = insights["intern_summary"]["primary_strength_driver"]
        self.assertIn("Output Score", strength)

    def test_empty_source_fallbacks_do_not_hallucinate(self) -> None:
        empty_attr = {
            "output_attribution": {"by_class": [], "by_adjustment": []},
            "efficiency_attribution": {"largest_overruns": [], "largest_underruns": []},
            "accuracy_attribution": {
                "top_error_drivers": [],
                "by_severity": [
                    {"severity": "minor", "weighted_error_impact": 0.0},
                    {"severity": "major", "weighted_error_impact": 0.0},
                ],
            },
            "contribution_attribution": {
                "positive_by_type": [],
                "negative_by_type": [],
                "raw_positive_effect": 0.0,
                "raw_negative_effect": 0.0,
            },
        }
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=2.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                total_weighted_errors=0.0,
                positive_flags=0,
                negative_flags=0,
                attribution_overrides=empty_attr,
                task_ids=["T201"],
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=2.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                total_weighted_errors=0.0,
                positive_flags=0,
                negative_flags=0,
                attribution_overrides=copy.deepcopy(empty_attr),
                task_ids=["T202"],
            ),
        }
        insights = build_diagnostic_insights(results, "INT001")
        expl = insights["attribution_explanations"]
        self.assertEqual(
            "Output attribution has no class or adjustment drivers to report.",
            expl["output_explanation"],
        )
        self.assertEqual(
            "Efficiency attribution shows no concentrated overrun or underrun driver.",
            expl["efficiency_explanation"],
        )
        self.assertEqual(
            "Accuracy attribution shows no weighted error burden.",
            expl["accuracy_explanation"],
        )
        self.assertEqual(
            "Contribution attribution shows no dominant positive or negative flag concentration.",
            expl["contribution_explanation"],
        )

    def test_accuracy_equal_major_minor_does_not_claim_major_dominance(self) -> None:
        attr = {
            "accuracy_attribution": {
                "top_error_drivers": [
                    {"task_id": "T301", "task_class": "C", "weighted_errors": 1.5}
                ],
                "by_severity": [
                    {"severity": "minor", "weighted_error_impact": 1.0},
                    {"severity": "major", "weighted_error_impact": 1.0},
                ],
            }
        }
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=1.0,
                performance_index=0.7,
                output_score=1.0,
                efficiency_score=1.2,
                accuracy_score=0.6,
                contribution_modifier=1.1,
                total_weighted_errors=1.5,
                attribution_overrides=attr,
                task_ids=["T301"],
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=2.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
        }
        insights = build_diagnostic_insights(results, "INT001")
        text = insights["attribution_explanations"]["accuracy_explanation"]
        self.assertNotIn("major-error concentration", text)
        self.assertIn("T301", text)

    def test_deterministic_output_same_input(self) -> None:
        results = run_scoring()
        intern_id = sorted(results.keys())[0]
        a = build_diagnostic_insights(results, intern_id)
        b = build_diagnostic_insights(results, intern_id)
        self.assertEqual(a, b)

    def test_single_clear_weakness_identified(self) -> None:
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=0.5,
                performance_index=0.5,
                output_score=1.0,
                efficiency_score=0.5,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.5,
                performance_index=0.9,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT003": _make_fake_result(
                "INT003",
                final_score=1.5,
                performance_index=0.9,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
        }
        insights = build_diagnostic_insights(results, "INT001")
        self.assertIn("Efficiency Score trails peer mean", insights["intern_summary"]["primary_weakness_driver"])


if __name__ == "__main__":
    unittest.main()
