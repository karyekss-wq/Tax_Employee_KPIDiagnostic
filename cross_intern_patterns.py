from __future__ import annotations

from collections import defaultdict
from typing import Any

from diagnostic_insights import build_diagnostic_insights


REQUIRED_PATTERN_FIELDS = [
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


def classify_pattern_scope(frequency: float) -> str:
    if frequency >= 0.50:
        return "systemic"
    if frequency >= 0.25:
        return "emerging"
    return "isolated"


def severity_from_frequency(frequency: float) -> str:
    if frequency >= 0.50:
        return "high"
    if frequency >= 0.25:
        return "moderate"
    if frequency > 0:
        return "low"
    return "neutral"


def collect_all_normalized_insights(results_by_intern: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for intern_id in sorted(results_by_intern.keys()):
        insights = build_diagnostic_insights(results_by_intern, intern_id)
        for record in insights["normalized_insights"]:
            merged = dict(record)
            merged["intern_id"] = str(intern_id)
            records.append(merged)
    return records


def _build_pattern(
    *,
    pattern_key: str,
    pattern_type: str,
    metric_source: str,
    direction: str,
    records: list[dict[str, Any]],
    total_interns: int,
    supporting_reference: str,
    message: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intern_ids = sorted({str(record["intern_id"]) for record in records})
    intern_count = len(intern_ids)
    frequency = float(intern_count / total_interns) if total_interns else 0.0

    evidence_values = [float(record.get("evidence_value", 0.0)) for record in records]
    evidence_value = float(sum(evidence_values) / len(evidence_values)) if evidence_values else 0.0
    evidence_unit = records[0].get("evidence_unit", "none") if records else "none"

    pattern = {
        "pattern_key": pattern_key,
        "pattern_type": pattern_type,
        "metric_source": metric_source,
        "direction": direction,
        "severity": severity_from_frequency(frequency),
        "scope_classification": classify_pattern_scope(frequency),
        "intern_count": intern_count,
        "total_interns": total_interns,
        "frequency": frequency,
        "supporting_reference": supporting_reference,
        "message": message,
        "sample_intern_ids": intern_ids,
        "evidence_value": evidence_value,
        "evidence_unit": evidence_unit,
    }
    if extra_fields:
        pattern.update(extra_fields)
    return pattern


def build_recurring_driver_patterns(
    normalized_records: list[dict[str, Any]], total_interns: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    target_keys = {
        "primary_strength_driver",
        "primary_weakness_driver",
        "dominant_final_score_driver",
    }
    for record in normalized_records:
        if record.get("insight_key") in target_keys:
            key = (
                str(record["insight_key"]),
                str(record.get("metric_source", "")),
                str(record.get("direction", "neutral")),
            )
            grouped[key].append(record)

    patterns: list[dict[str, Any]] = []
    for (insight_key, metric_source, direction), records in grouped.items():
        if insight_key == "primary_strength_driver":
            pattern_type = "recurring_strength"
            message = (
                f"{metric_source} recurs as primary strength in "
                f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
            )
        elif insight_key == "primary_weakness_driver":
            pattern_type = "recurring_weakness"
            message = (
                f"{metric_source} recurs as primary weakness in "
                f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
            )
        else:
            pattern_type = "recurring_driver"
            message = (
                f"{metric_source} recurs as dominant final-score driver in "
                f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
            )

        patterns.append(
            _build_pattern(
                pattern_key=f"recurring_{insight_key}_{metric_source}_{direction}",
                pattern_type=pattern_type,
                metric_source=metric_source,
                direction=direction,
                records=records,
                total_interns=total_interns,
                supporting_reference="normalized_insights",
                message=message,
                extra_fields={
                    "insight_key": insight_key,
                    "related_insight_keys": [insight_key],
                },
            )
        )

    return patterns


def build_recurring_positioning_patterns(
    normalized_records: list[dict[str, Any]], total_interns: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in normalized_records:
        key_name = str(record.get("insight_key", ""))
        direction = str(record.get("direction", "neutral"))
        if key_name in {"final_score_positioning", "performance_index_positioning"} and direction in {
            "support",
            "drag",
        }:
            key = (
                key_name,
                str(record.get("metric_source", "")),
                direction,
            )
            grouped[key].append(record)

    patterns: list[dict[str, Any]] = []
    for (insight_key, metric_source, direction), records in grouped.items():
        pattern_type = (
            "recurring_positioning_support" if direction == "support" else "recurring_positioning_drag"
        )
        message = (
            f"{metric_source} positioning shows repeated {direction} signals in "
            f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
        )

        patterns.append(
            _build_pattern(
                pattern_key=f"recurring_{insight_key}_{direction}",
                pattern_type=pattern_type,
                metric_source=metric_source,
                direction=direction,
                records=records,
                total_interns=total_interns,
                supporting_reference="cross_intern_comparison",
                message=message,
                extra_fields={"insight_key": insight_key, "related_insight_keys": [insight_key]},
            )
        )

    return patterns


def build_recurring_attribution_patterns(
    normalized_records: list[dict[str, Any]], total_interns: int
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []

    attribution_records = [
        record
        for record in normalized_records
        if record.get("insight_type") == "attribution" and record.get("direction") in {"support", "drag"}
    ]

    # Base recurring attribution support/drag by insight_key + source.
    grouped_base: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in attribution_records:
        grouped_base[
            (
                str(record.get("insight_key", "")),
                str(record.get("metric_source", "")),
                str(record.get("direction", "neutral")),
            )
        ].append(record)

    for (insight_key, metric_source, direction), records in grouped_base.items():
        pattern_type = "recurring_attribution_support" if direction == "support" else "recurring_attribution_drag"
        message = (
            f"{metric_source} {direction} attribution recurs in "
            f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
        )
        patterns.append(
            _build_pattern(
                pattern_key=f"recurring_{insight_key}_{metric_source}_{direction}",
                pattern_type=pattern_type,
                metric_source=metric_source,
                direction=direction,
                records=records,
                total_interns=total_interns,
                supporting_reference="normalized_insights",
                message=message,
                extra_fields={"insight_key": insight_key, "related_insight_keys": [insight_key]},
            )
        )

    # Repeated task class patterns.
    grouped_class: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in attribution_records:
        if record.get("related_task_class"):
            grouped_class[
                (
                    str(record["related_task_class"]),
                    str(record.get("metric_source", "")),
                    str(record.get("direction", "neutral")),
                )
            ].append(record)

    for (task_class, metric_source, direction), records in grouped_class.items():
        message = (
            f"Task class {task_class} recurs as a {direction} attribution source in "
            f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
        )
        patterns.append(
            _build_pattern(
                pattern_key=f"recurring_task_class_{task_class}_{metric_source}_{direction}",
                pattern_type="recurring_task_class_pattern",
                metric_source=metric_source,
                direction=direction,
                records=records,
                total_interns=total_interns,
                supporting_reference="normalized_insights",
                message=message,
                extra_fields={"task_class": task_class},
            )
        )

    # Repeated flag type patterns.
    grouped_flag: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in attribution_records:
        if record.get("related_flag_type"):
            grouped_flag[
                (
                    str(record["related_flag_type"]),
                    str(record.get("direction", "neutral")),
                )
            ].append(record)

    for (flag_type, direction), records in grouped_flag.items():
        message = (
            f"Flag type {flag_type} recurs as a {direction} contribution source in "
            f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
        )
        patterns.append(
            _build_pattern(
                pattern_key=f"recurring_flag_{flag_type}_{direction}",
                pattern_type="recurring_flag_pattern",
                metric_source="contribution_attribution",
                direction=direction,
                records=records,
                total_interns=total_interns,
                supporting_reference="contribution_attribution",
                message=message,
                extra_fields={"flag_type": flag_type},
            )
        )

    # Repeated adjustment code patterns.
    grouped_adjustment: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in attribution_records:
        if record.get("related_adjustment_code"):
            grouped_adjustment[
                (
                    str(record["related_adjustment_code"]),
                    str(record.get("direction", "neutral")),
                )
            ].append(record)

    for (adjustment_code, direction), records in grouped_adjustment.items():
        message = (
            f"Adjustment {adjustment_code} recurs as a {direction} output source in "
            f"{len({r['intern_id'] for r in records})} of {total_interns} interns."
        )
        patterns.append(
            _build_pattern(
                pattern_key=f"recurring_adjustment_{adjustment_code}_{direction}",
                pattern_type="recurring_adjustment_pattern",
                metric_source="output_attribution",
                direction=direction,
                records=records,
                total_interns=total_interns,
                supporting_reference="output_attribution",
                message=message,
                extra_fields={"adjustment_code": adjustment_code},
            )
        )

    return patterns


def _sort_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        patterns,
        key=lambda pattern: (
            -float(pattern["frequency"]),
            -int(pattern["intern_count"]),
            str(pattern["pattern_type"]),
            str(pattern["metric_source"]),
            str(pattern["pattern_key"]),
        ),
    )


def build_cross_intern_patterns(results_by_intern: dict[str, Any]) -> dict[str, Any]:
    total_interns = len(results_by_intern)
    normalized_records = collect_all_normalized_insights(results_by_intern)

    patterns: list[dict[str, Any]] = []
    patterns.extend(build_recurring_driver_patterns(normalized_records, total_interns))
    patterns.extend(build_recurring_positioning_patterns(normalized_records, total_interns))
    patterns.extend(build_recurring_attribution_patterns(normalized_records, total_interns))

    patterns = _sort_patterns(patterns)

    for pattern in patterns:
        for field in REQUIRED_PATTERN_FIELDS:
            if field not in pattern:
                raise ValueError(f"Pattern missing required field '{field}': {pattern}")

    summary = {
        "total_interns": total_interns,
        "total_patterns": len(patterns),
        "systemic_count": sum(1 for p in patterns if p["scope_classification"] == "systemic"),
        "emerging_count": sum(1 for p in patterns if p["scope_classification"] == "emerging"),
        "isolated_count": sum(1 for p in patterns if p["scope_classification"] == "isolated"),
    }

    return {
        "system_patterns": patterns,
        "pattern_summary": summary,
    }
