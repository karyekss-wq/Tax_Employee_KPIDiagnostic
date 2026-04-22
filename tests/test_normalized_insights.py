from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass

import pandas as pd

from diagnostic_insights import (
    COMPONENT_TIE_BREAK_ORDER,
    build_diagnostic_insights,
    get_metric_peer_context,
)
from normalized_insights import NORMALIZED_INSIGHT_ORDER
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
                    "output_contribution": 2.4,
                }
            ],
            "by_adjustment": [
                {"adjustment_code": "investments", "output_effect": 0.7}
            ],
        },
        "efficiency_attribution": {
            "largest_overruns": [
                {"task_id": "T901", "task_class": "B", "overrun_hours": 2.2}
            ],
            "largest_underruns": [
                {"task_id": "T902", "task_class": "A", "underrun_hours": 1.0}
            ],
        },
        "accuracy_attribution": {
            "top_error_drivers": [
                {"task_id": "T901", "task_class": "B", "weighted_errors": 1.2}
            ],
            "by_severity": [
                {"severity": "minor", "weighted_error_impact": 0.4},
                {"severity": "major", "weighted_error_impact": 1.1},
            ],
        },
        "contribution_attribution": {
            "positive_by_type": [
                {"flag_type": "helped_peer", "modifier_effect": 0.03}
            ],
            "negative_by_type": [
                {"flag_type": "rework_requested", "modifier_effect": -0.04}
            ],
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
        task_metrics=_base_task_metrics(task_ids or ["T901", "T902"], intern_id),
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


def _record_by_key(records: list[dict], key: str) -> dict:
    return next(record for record in records if record["insight_key"] == key)


class NormalizedInsightsTests(unittest.TestCase):
    def test_live_data_schema_and_order(self) -> None:
        results = run_scoring()
        intern_id = sorted(results.keys())[0]
        records = build_diagnostic_insights(results, intern_id)["normalized_insights"]

        self.assertEqual(NORMALIZED_INSIGHT_ORDER, [record["insight_key"] for record in records])

        required_fields = [
            "intern_id",
            "insight_key",
            "insight_type",
            "metric_source",
            "direction",
            "severity",
            "evidence_value",
            "evidence_unit",
            "evidence_label",
            "supporting_reference",
            "message",
        ]
        for record in records:
            for field in required_fields:
                self.assertIn(field, record)
            self.assertEqual(intern_id, record["intern_id"])

    def test_strength_and_weakness_metric_mapping_matches_peer_gap_rule(self) -> None:
        results = run_scoring()
        from diagnostic_insights import build_cross_intern_comparison

        comparison = build_cross_intern_comparison(results)
        for intern_id in sorted(results.keys()):
            records = build_diagnostic_insights(results, intern_id)["normalized_insights"]
            strength = _record_by_key(records, "primary_strength_driver")
            weakness = _record_by_key(records, "primary_weakness_driver")

            contexts = {
                metric: get_metric_peer_context(comparison, intern_id, metric)
                for metric in COMPONENT_TIE_BREAK_ORDER
            }
            gaps = {metric: contexts[metric]["peer_gap"] for metric in COMPONENT_TIE_BREAK_ORDER}

            positive_candidates = [m for m in COMPONENT_TIE_BREAK_ORDER if gaps[m] > 0]
            negative_candidates = [m for m in COMPONENT_TIE_BREAK_ORDER if gaps[m] < 0]

            if positive_candidates:
                expected_strength_metric = max(
                    positive_candidates,
                    key=lambda m: (gaps[m], -COMPONENT_TIE_BREAK_ORDER.index(m)),
                )
                self.assertEqual(expected_strength_metric, strength["metric_source"])
                self.assertAlmostEqual(gaps[expected_strength_metric], float(strength["evidence_value"]), places=6)
            else:
                self.assertEqual("neutral", strength["direction"])

            if negative_candidates:
                expected_weakness_metric = min(
                    negative_candidates,
                    key=lambda m: (gaps[m], COMPONENT_TIE_BREAK_ORDER.index(m)),
                )
                self.assertEqual(expected_weakness_metric, weakness["metric_source"])
                self.assertAlmostEqual(gaps[expected_weakness_metric], float(weakness["evidence_value"]), places=6)
            else:
                self.assertEqual("neutral", weakness["direction"])

    def test_positioning_records_match_rank_and_peer_mean(self) -> None:
        results = run_scoring()
        from diagnostic_insights import build_cross_intern_comparison

        comparison = build_cross_intern_comparison(results)
        for intern_id in sorted(results.keys()):
            records = build_diagnostic_insights(results, intern_id)["normalized_insights"]
            final_pos = _record_by_key(records, "final_score_positioning")
            perf_pos = _record_by_key(records, "performance_index_positioning")

            final_ctx = get_metric_peer_context(comparison, intern_id, "final_score")
            perf_ctx = get_metric_peer_context(comparison, intern_id, "performance_index")

            self.assertEqual(final_ctx["rank"], final_pos["rank"])
            self.assertEqual(perf_ctx["rank"], perf_pos["rank"])
            self.assertAlmostEqual(final_ctx["peer_mean"], float(final_pos["peer_mean"]), places=6)
            self.assertAlmostEqual(perf_ctx["peer_mean"], float(perf_pos["peer_mean"]), places=6)
            self.assertAlmostEqual(final_ctx["peer_gap"], float(final_pos["evidence_value"]), places=6)
            self.assertAlmostEqual(perf_ctx["peer_gap"], float(perf_pos["evidence_value"]), places=6)

    def test_attribution_related_fields_exist_in_payload(self) -> None:
        results = run_scoring()
        for intern_id in sorted(results.keys()):
            result = results[intern_id]
            records = build_diagnostic_insights(results, intern_id)["normalized_insights"]

            task_ids = set(result.task_metrics["task_id"].astype(str))
            task_classes = set(result.task_metrics.get("task_class", pd.Series(dtype=str)).astype(str))
            pos_flags = {row.get("flag_type") for row in result.attribution["contribution_attribution"].get("positive_by_type", [])}
            neg_flags = {row.get("flag_type") for row in result.attribution["contribution_attribution"].get("negative_by_type", [])}
            flag_types = {str(x) for x in (pos_flags | neg_flags) if x is not None}

            for record in records:
                if "related_task_id" in record and record["related_task_id"] is not None:
                    self.assertIn(str(record["related_task_id"]), task_ids)
                if "related_task_class" in record and record["related_task_class"] is not None and len(task_classes) > 0:
                    self.assertIn(str(record["related_task_class"]), task_classes)
                if "related_flag_type" in record and record["related_flag_type"] is not None and len(flag_types) > 0:
                    self.assertIn(str(record["related_flag_type"]), flag_types)

    def test_empty_source_fallbacks_are_neutral_and_non_hallucinatory(self) -> None:
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
                attribution_overrides=empty_attr,
                task_ids=["T1001"],
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
                task_ids=["T1002"],
            ),
        }

        records = build_diagnostic_insights(results, "INT001")["normalized_insights"]
        self.assertEqual("neutral", _record_by_key(records, "output_explanation")["direction"])
        self.assertEqual("neutral", _record_by_key(records, "efficiency_explanation")["direction"])
        self.assertEqual("support", _record_by_key(records, "accuracy_explanation")["direction"])
        self.assertEqual("neutral", _record_by_key(records, "contribution_explanation")["direction"])

    def test_equal_major_minor_uses_non_major_dominance_path(self) -> None:
        attr = {
            "accuracy_attribution": {
                "top_error_drivers": [
                    {"task_id": "T2001", "task_class": "C", "weighted_errors": 1.3}
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
                final_score=0.9,
                performance_index=0.7,
                output_score=1.0,
                efficiency_score=1.1,
                accuracy_score=0.7,
                contribution_modifier=1.0,
                total_weighted_errors=1.3,
                attribution_overrides=attr,
                task_ids=["T2001"],
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.2,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
            ),
        }
        record = _record_by_key(
            build_diagnostic_insights(results, "INT001")["normalized_insights"],
            "accuracy_explanation",
        )
        self.assertEqual("weighted_errors", record["evidence_unit"])
        self.assertEqual("T2001", record["related_task_id"])

    def test_deterministic_output_same_input(self) -> None:
        results = run_scoring()
        intern_id = sorted(results.keys())[0]
        a = build_diagnostic_insights(results, intern_id)["normalized_insights"]
        b = build_diagnostic_insights(results, intern_id)["normalized_insights"]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
