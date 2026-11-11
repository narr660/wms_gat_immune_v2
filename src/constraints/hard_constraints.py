
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable


@dataclass(frozen=True)
class ConstraintRule:
    rule_id: str
    source: str
    source_section: str
    risk_reason: str
    hard_constraint: bool
    verified: bool = False
    rule_type: str = "synthetic"


@dataclass
class RuleBinding:
    rule: ConstraintRule
    predicate: Callable[[Dict, Dict], Optional[bool]]


def evaluate_hard_constraints(
    a: Dict,
    b: Dict,
    bindings: Optional[List[RuleBinding]] = None,
    enable_synthetic: bool = False,
) -> Dict:

    matched = []
    unknown = False
    applicable_count = 0

    if not bindings:
        return {
            "hard_forbidden": False,
            "matched_rules": [],
            "evaluation_status": "unknown",
        }

    for binding in bindings:
        rule = binding.rule

        if rule.rule_type not in (
            "regulatory", "safety_principle", "synthetic"
        ):
            raise ValueError(
                f"Unknown rule_type: {rule.rule_type}"
            )

        if rule.rule_type == "synthetic" and not enable_synthetic:
            continue

        if rule.rule_type in ("regulatory", "safety_principle"):
            if not rule.verified:
                unknown = True
                continue

        applicable_count += 1
        result = binding.predicate(a, b)

        if result is None:
            unknown = True
            continue

        if result is True:
            matched.append(asdict(rule))

    forbidden = any(
        rule["hard_constraint"] for rule in matched
    )

    if forbidden:
        status = "forbidden"
    elif unknown or applicable_count == 0:
        status = "unknown"
    else:
        status = "no_prohibition_found"

    return {
        "hard_forbidden": forbidden,
        "matched_rules": matched,
        "evaluation_status": status,
    }
