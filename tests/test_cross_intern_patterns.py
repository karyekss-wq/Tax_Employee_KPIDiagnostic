from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass

import pandas as pd

from cross_intern_patterns import (
    build_cross_intern_patterns,
    classify_pattern_scope,
    collect_all_normalized_insights,
    severity_from_frequency,
)
from scoring import run_scoring


@dataclass
class FakeResult:
    task_metrics: pd.DataFrame
    summary: dict
    attribution: dict


def _base_task_metrics(task_ids: list[str], intern_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": task_ids,
            "intern_id": [intern_id] * len(task_ids),
            "task_class": ["B"] * len(task_ids),
        }
    )


def _base_attribution() -> dict:
    return {
        "output_attribution": {
            "by_class": [
                {"task_class": "B", "class_name": "Business", "output_contribution": 2.0}
            ],
            "by_adjustment": [
                {"adjustment_code": "investments", "output_effect": 0.6}
            ],
        },
        "efficiency_attribution": {
            "largest_overruns": [
                {"task_id": "T700", "task_class": "B", "overrun_hours": 2.0}
            ],
            "largest_underruns": [
                {"task_id": "T701", "task_class": "A", "underrun_hours": 1.0}
            ],
        },
        "accuracy_attribution": {
            "top_error_drivers": [
                {"task_id": "T700", "task_class": "B", "weighted_errors": 1.3}
            ],
            "by_severity": [
                {"severity": "minor", "weighted_error_impact": 0.4},
                {"severity": "major", "weighted_error_impact": 1.2},
            ],
        },
        "contribution_attribution": {
            "positive_by_type": [{"flag_type": "helped_peer", "modifier_effect": 0.03}],
            "negative_by_type": [{"flag_type": "rework_requested", "modifier_effect": -0.04}],
            "raw_positive_effect": 0.03,
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
        task_metrics=_base_task_metrics(task_ids or ["T700", "T701"], intern_id),
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


class CrossInternPatternTests(unittest.TestCase):
    def test_live_data_patterns_build_and_schema(self) -> None:
        payload = build_cross_intern_patterns(run_scoring())
        self.assertIn("system_patterns", payload)
        self.assertIn("pattern_summary", payload)
        patterns = payload["system_patterns"]
        summary = payload["pattern_summary"]

        required = [
            "pattern_key",
            "pattern_type",
            "metric_source",
            "direction",
            "severity",
            "intern_count",
            "total_interns",
            "frequency",
            "supporting_reference",
            "message",
        ]
        for pattern in patterns:
            for field in required:
                self.assertIn(field, pattern)
            self.assertAlmostEqual(
                pattern["intern_count"] / pattern["total_interns"], pattern["frequency"], places=12
            )

        self.assertEqual(summary["total_patterns"], len(patterns))

    def test_severity_and_scope_threshold_mapping(self) -> None:
        self.assertEqual("high", severity_from_frequency(0.50))
        self.assertEqual("moderate", severity_from_frequency(0.25))
        self.assertEqual("low", severity_from_frequency(0.10))
        self.assertEqual("neutral", severity_from_frequency(0.0))

        self.assertEqual("systemic", classify_pattern_scope(0.50))
        self.assertEqual("emerging", classify_pattern_scope(0.25))
        self.assertEqual("isolated", classify_pattern_scope(0.10))

    def test_output_is_deterministic_for_same_input(self) -> None:
        results = run_scoring()
        a = build_cross_intern_patterns(results)
        b = build_cross_intern_patterns(results)
        self.assertEqual(a, b)

    def test_pattern_ordering_is_deterministic_with_tied_frequencies(self) -> None:
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=1.0,
                performance_index=0.8,
                output_score=2.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=2.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
            "INT003": _make_fake_result(
                "INT003",
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=2.0,
                contribution_modifier=1.0,
            ),
            "INT004": _make_fake_result(
                "INT004",
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=2.0,
            ),
        }
        payload = build_cross_intern_patterns(results)
        patterns = payload["system_patterns"]

        sorted_again = sorted(
            patterns,
            key=lambda pattern: (
                -float(pattern["frequency"]),
                -int(pattern["intern_count"]),
                str(pattern["pattern_type"]),
                str(pattern["metric_source"]),
                str(pattern["pattern_key"]),
            ),
        )
        self.assertEqual(patterns, sorted_again)

    def test_references_exist_for_task_class_flag_adjustment_patterns(self) -> None:
        results = run_scoring()
        payload = build_cross_intern_patterns(results)

        normalized_records = collect_all_normalized_insights(results)
        task_classes = {str(r.get("related_task_class")) for r in normalized_records if r.get("related_task_class")}
        flag_types = {str(r.get("related_flag_type")) for r in normalized_records if r.get("related_flag_type")}
        adjustment_codes = {
            str(r.get("related_adjustment_code")) for r in normalized_records if r.get("related_adjustment_code")
        }

        for pattern in payload["system_patterns"]:
            if "task_class" in pattern:
                self.assertIn(str(pattern["task_class"]), task_classes)
            if "flag_type" in pattern:
                self.assertIn(str(pattern["flag_type"]), flag_types)
            if "adjustment_code" in pattern:
                self.assertIn(str(pattern["adjustment_code"]), adjustment_codes)

    def test_no_hallucinated_source_patterns_when_related_fields_absent(self) -> None:
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
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                total_weighted_errors=0.0,
                positive_flags=0,
                negative_flags=0,
                attribution_overrides=copy.deepcopy(empty_attr),
                task_ids=["T801"],
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                total_weighted_errors=0.0,
                positive_flags=0,
                negative_flags=0,
                attribution_overrides=copy.deepcopy(empty_attr),
                task_ids=["T802"],
            ),
        }

        payload = build_cross_intern_patterns(results)
        keys = {pattern["pattern_type"] for pattern in payload["system_patterns"]}
        self.assertNotIn("recurring_task_class_pattern", keys)
        self.assertNotIn("recurring_flag_pattern", keys)
        self.assertNotIn("recurring_adjustment_pattern", keys)

    def test_single_intern_match_pattern_frequency(self) -> None:
        results = run_scoring()
        payload = build_cross_intern_patterns(results)
        total = payload["pattern_summary"]["total_interns"]
        self.assertGreaterEqual(total, 1)

        found_single = any(pattern["intern_count"] == 1 for pattern in payload["system_patterns"])
        self.assertTrue(found_single)
        for pattern in payload["system_patterns"]:
            if pattern["intern_count"] == 1:
                self.assertAlmostEqual(1 / total, pattern["frequency"], places=12)


if __name__ == "__main__":
    unittest.main()
