"""Unit tests for the SOP compliance DSL.

Pin the rule grammar + the evaluator semantics so future changes to either
break loudly. No I/O — these are pure-function tests, run in milliseconds.
"""
from __future__ import annotations

import pytest

from app.sop import Event, Rule, check_rules, evaluate, parse_rules


# ---------- payload fixtures ----------


def _payload_with(
    interactions: list[tuple[str, int, int]] = (),
    motion: list[tuple[str, str, int, int]] = (),
) -> dict:
    """Build a minimal payload. `interactions` is [(class, start, end), ...];
    `motion` is [(class, state, start, end), ...]. One object per class."""
    by_class: dict[str, dict] = {}
    for cls, s, e in interactions:
        obj = by_class.setdefault(cls, {"class": cls, "interactions": [], "motion_history": []})
        obj["interactions"].append({"interacted_by_person": 0, "frame_start": s, "frame_end": e})
    for cls, state, s, e in motion:
        obj = by_class.setdefault(cls, {"class": cls, "interactions": [], "motion_history": []})
        obj["motion_history"].append({"frame_range": [s, e], "state": state})
    return {"objectsDetected": list(by_class.values())}


# ---------- parser ----------


def test_parse_required_interaction():
    rules = parse_rules("REQUIRED interaction(person, spectrophotometer)")
    assert rules == [
        Rule(
            raw="REQUIRED interaction(person, spectrophotometer)",
            op="REQUIRED",
            left=Event(kind="interaction", cls="spectrophotometer"),
        )
    ]


def test_parse_before_motion_interaction():
    rules = parse_rules(
        "motion(cable, moving) BEFORE interaction(person, spectrophotometer)"
    )
    assert rules[0].op == "BEFORE"
    assert rules[0].left == Event(kind="motion", cls="cable", state="moving")
    assert rules[0].right == Event(kind="interaction", cls="spectrophotometer")


def test_parse_skips_comments_and_blanks():
    rules = parse_rules(
        """
        # this is a comment

        REQUIRED interaction(person, *)
        # trailing comment
        """
    )
    assert len(rules) == 1
    assert rules[0].left.cls == "*"


def test_parse_unknown_op_raises_with_line_number():
    text = "REQUIRED interaction(person, x)\nGIBBERISH foo BETWEEN bar"
    with pytest.raises(ValueError, match="line 2"):
        parse_rules(text)


def test_parse_malformed_event_raises():
    with pytest.raises(ValueError, match="could not parse event"):
        parse_rules("REQUIRED something_weird(foo)")


# ---------- evaluator: REQUIRED / FORBIDDEN ----------


def test_required_passes_when_event_present():
    payload = _payload_with(interactions=[("spectrophotometer", 10, 20)])
    rules = parse_rules("REQUIRED interaction(person, spectrophotometer)")
    [r] = check_rules(rules, payload)
    assert r.passed


def test_required_fails_when_event_absent():
    payload = _payload_with(interactions=[("bottle", 0, 5)])
    rules = parse_rules("REQUIRED interaction(person, spectrophotometer)")
    [r] = check_rules(rules, payload)
    assert not r.passed
    assert "no matching" in r.detail


def test_forbidden_passes_when_event_absent():
    payload = _payload_with()
    rules = parse_rules("FORBIDDEN interaction(person, cable)")
    [r] = check_rules(rules, payload)
    assert r.passed


def test_forbidden_fails_when_event_present():
    payload = _payload_with(interactions=[("cable", 0, 10)])
    rules = parse_rules("FORBIDDEN interaction(person, cable)")
    [r] = check_rules(rules, payload)
    assert not r.passed


def test_wildcard_class_matches_any_object():
    payload = _payload_with(interactions=[("bottle", 0, 5)])
    rules = parse_rules("REQUIRED interaction(person, *)")
    [r] = check_rules(rules, payload)
    assert r.passed


# ---------- evaluator: ordering ----------


def test_before_passes_when_left_ends_before_right_starts():
    payload = _payload_with(
        interactions=[("cable", 0, 30), ("spectrophotometer", 60, 100)]
    )
    rules = parse_rules(
        "interaction(person, cable) BEFORE interaction(person, spectrophotometer)"
    )
    [r] = check_rules(rules, payload)
    assert r.passed


def test_before_fails_when_intervals_overlap():
    payload = _payload_with(
        interactions=[("cable", 0, 80), ("spectrophotometer", 60, 100)]
    )
    rules = parse_rules(
        "interaction(person, cable) BEFORE interaction(person, spectrophotometer)"
    )
    [r] = check_rules(rules, payload)
    assert not r.passed


def test_after_is_before_swapped():
    payload = _payload_with(
        interactions=[("cable", 60, 100), ("spectrophotometer", 0, 30)]
    )
    rules = parse_rules(
        "interaction(person, cable) AFTER interaction(person, spectrophotometer)"
    )
    [r] = check_rules(rules, payload)
    assert r.passed


def test_during_passes_when_left_contained_in_right():
    payload = _payload_with(
        interactions=[("cable", 30, 50)],
        motion=[("spectrophotometer", "stationary", 0, 191)],
    )
    rules = parse_rules(
        "interaction(person, cable) DURING motion(spectrophotometer, stationary)"
    )
    [r] = check_rules(rules, payload)
    assert r.passed


def test_during_fails_when_left_extends_outside_right():
    payload = _payload_with(
        interactions=[("cable", 0, 100)],
        motion=[("spectrophotometer", "stationary", 50, 80)],
    )
    rules = parse_rules(
        "interaction(person, cable) DURING motion(spectrophotometer, stationary)"
    )
    [r] = check_rules(rules, payload)
    assert not r.passed


def test_binary_op_fails_with_helpful_detail_when_precondition_missing():
    payload = _payload_with(interactions=[("cable", 0, 30)])
    rules = parse_rules(
        "interaction(person, cable) BEFORE interaction(person, spectrophotometer)"
    )
    [r] = check_rules(rules, payload)
    assert not r.passed
    assert "missing precondition" in r.detail


# ---------- top-level evaluate() ----------


def test_evaluate_returns_summary_and_per_rule():
    payload = _payload_with(
        interactions=[("spectrophotometer", 10, 20)],
    )
    text = """
    REQUIRED interaction(person, spectrophotometer)
    FORBIDDEN interaction(person, cable)
    """
    out = evaluate(text, payload)
    assert out["rules_total"] == 2
    assert out["rules_passed"] == 2
    assert out["all_passed"] is True
    assert len(out["results"]) == 2


def test_evaluate_marks_overall_failure_if_any_rule_fails():
    payload = _payload_with()
    out = evaluate("REQUIRED interaction(person, spectrophotometer)", payload)
    assert out["all_passed"] is False
    assert out["rules_failed"] == 1
