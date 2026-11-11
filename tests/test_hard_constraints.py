
from src.constraints.hard_constraints import (
    ConstraintRule,
    RuleBinding,
    evaluate_hard_constraints,
)


A = {"category": "화학품"}
B = {"category": "식품"}


def make_binding(rule_type, verified, matches=True):
    rule = ConstraintRule(
        rule_id="TEST-001",
        source="TEST_ONLY",
        source_section="TEST",
        risk_reason="테스트용 조건",
        hard_constraint=True,
        verified=verified,
        rule_type=rule_type,
    )
    return RuleBinding(
        rule=rule,
        predicate=lambda a, b: matches,
    )


def test_unverified_regulatory_rule():
    result = evaluate_hard_constraints(
        A, B,
        bindings=[make_binding("regulatory", False)],
    )
    assert result["hard_forbidden"] is False
    assert result["evaluation_status"] == "unknown"


def test_verified_regulatory_rule():
    result = evaluate_hard_constraints(
        A, B,
        bindings=[make_binding("regulatory", True)],
    )
    assert result["hard_forbidden"] is True
    assert result["evaluation_status"] == "forbidden"
    assert result["matched_rules"][0]["rule_id"] == "TEST-001"


def test_synthetic_rule_disabled():
    result = evaluate_hard_constraints(
        A, B,
        bindings=[make_binding("synthetic", False)],
    )
    assert result["hard_forbidden"] is False
    assert result["evaluation_status"] == "unknown"


def test_synthetic_rule_enabled():
    result = evaluate_hard_constraints(
        A, B,
        bindings=[make_binding("synthetic", False)],
        enable_synthetic=True,
    )
    assert result["hard_forbidden"] is True
    assert result["evaluation_status"] == "forbidden"


def test_no_rules():
    result = evaluate_hard_constraints(A, B)
    assert result["hard_forbidden"] is False
    assert result["evaluation_status"] == "unknown"


def test_non_matching_rule():
    result = evaluate_hard_constraints(
        A, B,
        bindings=[make_binding("regulatory", True, matches=False)],
    )
    assert result["hard_forbidden"] is False
    assert result["evaluation_status"] == "no_prohibition_found"


def test_unknown_predicate():
    binding = make_binding("regulatory", True)
    binding.predicate = lambda a, b: None

    result = evaluate_hard_constraints(A, B, bindings=[binding])
    assert result["evaluation_status"] == "unknown"
    assert result["hard_forbidden"] is False
