"""SOP compliance DSL — checks an objectsDetected/interactions trace
against a set of human-readable rules and returns pass/fail per rule.

This is the smallest thing that turns the witness output from a *trace*
into a *check*. The trace says what happened; the check says whether what
happened was right.

Rule grammar (one rule per line, # for comments, blank lines ignored):

    REQUIRED <event>
        Event must occur at least once.

    <event> BEFORE <event>
        First event's interval ends before second event's interval starts.

    <event> AFTER <event>
        Equivalent to "B BEFORE A".

    <event> DURING <event>
        First event's interval is fully contained inside the second's.

    FORBIDDEN <event>
        Event must NOT occur.

Event grammar:

    interaction(person, <class>)
        A person interacted with at least one object of <class>.
        <class> may be a literal label ("spectrophotometer") or "*" (any).

    motion(<class>, moving|stationary)
        At least one object of <class> entered the given motion state.

Examples:

    REQUIRED interaction(person, spectrophotometer)
    interaction(person, cable) BEFORE interaction(person, spectrophotometer)
    motion(spectrophotometer, stationary) DURING interaction(person, spectrophotometer)
    FORBIDDEN interaction(person, *)

This is intentionally tiny. A real production DSL would have variables,
quantifiers ("EVERY interaction"), counts, durations. v0.1 ships the
ordering primitives that compliance officers actually ask for first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional


# ---------- AST ----------


@dataclass(frozen=True)
class Event:
    """Either an interaction (a person touching an object class) or a
    motion (an object class entering a motion state). `cls` may be '*' to
    mean any class. `state` is only set on motion events."""

    kind: Literal["interaction", "motion"]
    cls: str
    state: Optional[Literal["moving", "stationary"]] = None


@dataclass(frozen=True)
class Rule:
    raw: str
    op: Literal["REQUIRED", "FORBIDDEN", "BEFORE", "AFTER", "DURING"]
    left: Event
    right: Optional[Event] = None  # only set for binary ops


# ---------- parser ----------


_INTERACTION_RE = re.compile(r"interaction\(\s*person\s*,\s*([A-Za-z0-9_*\- ]+?)\s*\)")
_MOTION_RE = re.compile(r"motion\(\s*([A-Za-z0-9_*\- ]+?)\s*,\s*(moving|stationary)\s*\)")


def _parse_event(s: str) -> Event:
    s = s.strip()
    m = _INTERACTION_RE.fullmatch(s)
    if m:
        return Event(kind="interaction", cls=m.group(1).strip())
    m = _MOTION_RE.fullmatch(s)
    if m:
        return Event(
            kind="motion", cls=m.group(1).strip(),
            state=m.group(2),  # type: ignore[arg-type]
        )
    raise ValueError(f"could not parse event: {s!r}")


def parse_rules(text: str) -> list[Rule]:
    """Parse the DSL text into a list of Rule objects. Raises ValueError
    with a line-numbered message on syntax errors."""
    rules: list[Rule] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rules.append(_parse_one(line))
        except ValueError as e:
            raise ValueError(f"line {lineno}: {e}") from None
    return rules


def _parse_one(line: str) -> Rule:
    upper = line.upper()
    if upper.startswith("REQUIRED "):
        return Rule(raw=line, op="REQUIRED", left=_parse_event(line[len("REQUIRED "):]))
    if upper.startswith("FORBIDDEN "):
        return Rule(raw=line, op="FORBIDDEN", left=_parse_event(line[len("FORBIDDEN "):]))
    for kw in (" BEFORE ", " AFTER ", " DURING "):
        idx = upper.find(kw)
        if idx == -1:
            continue
        left = _parse_event(line[:idx])
        right = _parse_event(line[idx + len(kw):])
        op = kw.strip()
        return Rule(raw=line, op=op, left=left, right=right)  # type: ignore[arg-type]
    raise ValueError(f"unrecognized rule: {line!r}")


# ---------- evaluator ----------


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def ends_before(self, other: "Interval") -> bool:
        return self.end < other.start

    def contained_in(self, other: "Interval") -> bool:
        return self.start >= other.start and self.end <= other.end


def _intervals_for(event: Event, payload: dict) -> list[Interval]:
    """Find every frame interval in the payload that matches this event."""
    out: list[Interval] = []
    for obj in payload.get("objectsDetected", []):
        if event.cls != "*" and obj.get("class") != event.cls:
            continue
        if event.kind == "interaction":
            for iv in obj.get("interactions", []):
                out.append(Interval(iv["frame_start"], iv["frame_end"]))
        else:  # motion
            for iv in obj.get("motion_history", []):
                if iv.get("state") != event.state:
                    continue
                fr = iv.get("frame_range", [0, 0])
                out.append(Interval(fr[0], fr[1]))
    return out


@dataclass(frozen=True)
class RuleResult:
    rule: str
    op: str
    passed: bool
    detail: str


def check_rules(rules: list[Rule], payload: dict) -> list[RuleResult]:
    """Evaluate every rule against the payload. Pure function — no I/O."""
    results: list[RuleResult] = []
    for r in rules:
        left = _intervals_for(r.left, payload)
        if r.op == "REQUIRED":
            ok = bool(left)
            results.append(RuleResult(
                rule=r.raw, op=r.op, passed=ok,
                detail=f"found {len(left)} matching interval(s)"
                if ok else "no matching interval found",
            ))
            continue
        if r.op == "FORBIDDEN":
            ok = not left
            results.append(RuleResult(
                rule=r.raw, op=r.op, passed=ok,
                detail="no matching interval (as required)"
                if ok else f"forbidden event occurred in {len(left)} interval(s)",
            ))
            continue
        # Binary ops require r.right.
        assert r.right is not None
        right = _intervals_for(r.right, payload)
        if not left or not right:
            results.append(RuleResult(
                rule=r.raw, op=r.op, passed=False,
                detail=(
                    f"missing precondition: "
                    f"left={len(left)} interval(s), right={len(right)} interval(s)"
                ),
            ))
            continue
        if r.op == "BEFORE":
            ok = any(li.ends_before(ri) for li in left for ri in right)
            results.append(RuleResult(
                rule=r.raw, op=r.op, passed=ok,
                detail="found a left interval ending before a right interval"
                if ok else "no left interval ends before any right interval",
            ))
        elif r.op == "AFTER":
            ok = any(ri.ends_before(li) for li in left for ri in right)
            results.append(RuleResult(
                rule=r.raw, op=r.op, passed=ok,
                detail="found a left interval starting after a right interval"
                if ok else "no left interval starts after any right interval",
            ))
        elif r.op == "DURING":
            ok = any(li.contained_in(ri) for li in left for ri in right)
            results.append(RuleResult(
                rule=r.raw, op=r.op, passed=ok,
                detail="found a left interval fully inside a right interval"
                if ok else "no left interval is fully contained in any right interval",
            ))
    return results


def evaluate(text: str, payload: dict) -> dict:
    """Top-level entry point: parse + evaluate, return a JSON-serializable
    summary. Used by the HTTP endpoint."""
    rules = parse_rules(text)
    results = check_rules(rules, payload)
    passed = sum(1 for r in results if r.passed)
    return {
        "rules_total": len(results),
        "rules_passed": passed,
        "rules_failed": len(results) - passed,
        "all_passed": passed == len(results),
        "results": [
            {"rule": r.rule, "op": r.op, "passed": r.passed, "detail": r.detail}
            for r in results
        ],
    }
