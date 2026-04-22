from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass

import pandas as pd

from manager_actions import (
    PRIORITY_ORDER,
    build_intern_manager_actions,
    build_manager_actions,
    build_team_manager_actions,
    deduplicate_actions,
)
from scoring import run_scoring


@dataclass
class FakeResult:
    task_metrics: pd.DataFrame
    summary: dict
    attribution: dict


def _base_task_metrics(task_ids: list[str], intern_id: str, task_class: str = "B") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": task_ids,
            "intern_id": [intern_id] * len(task_ids),
            "task_class": [task_class] * len(task_ids),
        }
    )


def _base_attribution() -> dict:
    return {
        "output_attribution": {
            "by_class": [
                {"task_class": "B", "class_name": "Business", "output_contribution": 2.2}
            ],
            "by_adjustment": [
                {"adjustment_code": "investments", "output_effect": 0.5}
            ],
        },
        "efficiency_attribution": {
            "largest_overruns": [
                {"task_id": "T501", "task_class": "B", "overrun_hours": 2.4}
            ],
            "largest_underruns": [
                {"task_id": "T502", "task_class": "A", "underrun_hours": 1.2}
            ],
        },
        "accuracy_attribution": {
            "top_error_drivers": [
                {"task_id": "T501", "task_class": "B", "weighted_errors": 1.6}
            ],
            "by_severity": [
                {"severity": "minor", "weighted_error_impact": 0.5},
                {"severity": "major", "weighted_error_impact": 1.4},
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
    task_class: str = "B",
) -> FakeResult:
    attribution = _base_attribution()
    if attribution_overrides:
        for key, value in attribution_overrides.items():
            attribution[key] = value

    return FakeResult(
        task_metrics=_base_task_metrics(task_ids or ["T501", "T502"], intern_id, task_class=task_class),
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


class ManagerActionTests(unittest.TestCase):
    def test_live_data_actions_build_and_schema(self) -> None:
        payload = build_manager_actions(run_scoring())
        self.assertIn("intern_actions", payload)
        self.assertIn("team_actions", payload)
        self.assertIn("action_summary", payload)

        required_fields = [
            "action_key",
            "action_type",
            "target_scope",
            "target_id",
            "priority_level",
            "evidence_sources",
            "trigger_type",
            "message",
            "rationale",
            "related_metric_source",
        ]
        all_actions = payload["intern_actions"] + payload["team_actions"]
        for action in all_actions:
            for field in required_fields:
                self.assertIn(field, action)
            self.assertTrue(action["evidence_sources"])  # no action without evidence

        summary = payload["action_summary"]
        self.assertEqual(summary["total_intern_actions"], len(payload["intern_actions"]))
        self.assertEqual(summary["total_team_actions"], len(payload["team_actions"]))
        self.assertEqual(
            summary["high_priority_count"]
            + summary["moderate_priority_count"]
            + summary["low_priority_count"],
            len(all_actions),
        )

    def test_deduplicate_actions(self) -> None:
        actions = [
            {
                "action_key": "a",
                "action_type": "watchlist",
                "target_scope": "intern",
                "target_id": "INT001",
                "priority_level": "low",
                "evidence_sources": ["normalized_insights"],
                "trigger_type": "isolated_weakness",
                "message": "x",
                "rationale": "y",
                "related_metric_source": "efficiency_score",
            },
            {
                "action_key": "a",
                "action_type": "watchlist",
                "target_scope": "intern",
                "target_id": "INT001",
                "priority_level": "low",
                "evidence_sources": ["normalized_insights"],
                "trigger_type": "isolated_weakness",
                "message": "x",
                "rationale": "y",
                "related_metric_source": "efficiency_score",
            },
        ]
        deduped = deduplicate_actions(actions)
        self.assertEqual(1, len(deduped))

    def test_ordering_is_deterministic(self) -> None:
        payload_a = build_manager_actions(run_scoring())
        payload_b = build_manager_actions(run_scoring())
        self.assertEqual(payload_a, payload_b)

        for key in ["intern_actions", "team_actions"]:
            actions = payload_a[key]
            sorted_again = sorted(
                actions,
                key=lambda action: (
                    PRIORITY_ORDER.get(str(action.get("priority_level", "low")), 99),
                    str(action.get("target_scope", "")),
                    str(action.get("action_type", "")),
                    str(action.get("target_id", "")),
                    str(action.get("action_key", "")),
                ),
            )
            self.assertEqual(actions, sorted_again)

    def test_target_scope_and_target_id_are_consistent(self) -> None:
        payload = build_manager_actions(run_scoring())

        for action in payload["intern_actions"]:
            self.assertEqual("intern", action["target_scope"])
            self.assertTrue(str(action["target_id"]).startswith("INT"))

        for action in payload["team_actions"]:
            self.assertIn(action["target_scope"], {"team", "system"})
            self.assertTrue(str(action["target_id"]))

    def test_no_hallucinated_optional_references(self) -> None:
        results = run_scoring()
        payload = build_manager_actions(results)

        task_classes = set()
        flag_types = set()
        adjustment_codes = set()
        for intern_id, result in results.items():
            task_classes.update(result.task_metrics["task_class"].astype(str).tolist())
            flag_types.update(
                str(row.get("flag_type"))
                for row in result.attribution["contribution_attribution"].get("positive_by_type", [])
            )
            flag_types.update(
                str(row.get("flag_type"))
                for row in result.attribution["contribution_attribution"].get("negative_by_type", [])
            )
            adjustment_codes.update(
                str(row.get("adjustment_code"))
                for row in result.attribution["output_attribution"].get("by_adjustment", [])
            )

        for action in payload["team_actions"] + payload["intern_actions"]:
            if action.get("task_class"):
                self.assertIn(str(action["task_class"]), task_classes)
            if action.get("flag_type"):
                self.assertIn(str(action["flag_type"]), flag_types)
            if action.get("adjustment_code"):
                self.assertIn(str(action["adjustment_code"]), adjustment_codes)

    def test_support_only_scenario_creates_preserve_action(self) -> None:
        empty_attr = {
            "efficiency_attribution": {"largest_overruns": [], "largest_underruns": []},
            "accuracy_attribution": {
                "top_error_drivers": [],
                "by_severity": [
                    {"severity": "minor", "weighted_error_impact": 0.0},
                    {"severity": "major", "weighted_error_impact": 0.0},
                ],
            },
            "contribution_attribution": {
                "positive_by_type": [{"flag_type": "helped_peer", "modifier_effect": 0.02}],
                "negative_by_type": [],
                "raw_positive_effect": 0.02,
                "raw_negative_effect": 0.0,
            },
        }
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=2.0,
                performance_index=0.95,
                output_score=2.0,
                efficiency_score=1.1,
                accuracy_score=1.0,
                contribution_modifier=1.02,
                total_weighted_errors=0.0,
                positive_flags=2,
                negative_flags=0,
                attribution_overrides=copy.deepcopy(empty_attr),
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.0,
                performance_index=0.80,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                total_weighted_errors=0.0,
                positive_flags=0,
                negative_flags=0,
                attribution_overrides=copy.deepcopy(empty_attr),
            ),
        }

        intern_actions = build_intern_manager_actions(results)
        preserve_actions = [a for a in intern_actions if a["action_type"] in {"preserve_strength", "recognition_signal"}]
        self.assertTrue(preserve_actions)

    def test_isolated_weakness_maps_to_low_watchlist(self) -> None:
        isolated_attr = {
            "efficiency_attribution": {"largest_overruns": [], "largest_underruns": []},
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
                performance_index=0.7,
                output_score=1.0,
                efficiency_score=0.99,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                attribution_overrides=copy.deepcopy(isolated_attr),
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.0,
                performance_index=0.7,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                attribution_overrides=copy.deepcopy(isolated_attr),
            ),
        }
        intern_actions = build_intern_manager_actions(results)
        watchlist = [a for a in intern_actions if a["action_type"] == "watchlist"]
        self.assertTrue(watchlist)
        self.assertTrue(all(a["priority_level"] == "low" for a in watchlist))

    def test_handles_empty_team_actions(self) -> None:
        neutral_attr = {
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

        # All interns neutral/equal -> no support/drag recurring team patterns should map to actions.
        results = {
            "INT001": _make_fake_result(
                "INT001",
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                positive_flags=0,
                negative_flags=0,
                total_weighted_errors=0.0,
                attribution_overrides=copy.deepcopy(neutral_attr),
            ),
            "INT002": _make_fake_result(
                "INT002",
                final_score=1.0,
                performance_index=0.8,
                output_score=1.0,
                efficiency_score=1.0,
                accuracy_score=1.0,
                contribution_modifier=1.0,
                positive_flags=0,
                negative_flags=0,
                total_weighted_errors=0.0,
                attribution_overrides=copy.deepcopy(neutral_attr),
            ),
        }
        team_actions = build_team_manager_actions(results)
        self.assertEqual([], team_actions)


if __name__ == "__main__":
    unittest.main()
