
from src.constraints.integrated_risk import assess_pair
from src.constraints.hard_constraints import ConstraintRule, RuleBinding


A = {
    "odor_profile": "release",
    "sealed": False,
    "category": "일반",
    "is_electronic": False,
    "hazard_type": "none",
    "hazard_level": 0.1,
    "leak_risk": 0.0,
    "weight_sim": 10.0,
    "max_supported_weight_sim": 100.0,
    "fragile": False,
}

B = {
    **A,
    "odor_profile": "absorb",
}


def make_binding(rule_type, verified):
    rule = ConstraintRule(
        rule_id="TEST-INT-001",
        source="TEST_ONLY",
        source_section="TEST",
        risk_reason="통합 판정 테스트",
        hard_constraint=True,
        verified=verified,
        rule_type=rule_type,
    )
    return RuleBinding(
        rule=rule,
        predicate=lambda a, b: True,
    )


def test_indirect_risk():
    result = assess_pair(A, B)
    assert result["relation_risks"]["odor"] == 3
    assert result["indirect_risk"] == 3
    assert result["final_risk"] == 3
    assert result["hard_forbidden"] is False
    assert result["constraint_status"] == "unknown"


def test_verified_hard_constraint():
    result = assess_pair(
        A, B,
        bindings=[make_binding("regulatory", True)],
    )
    assert result["final_risk"] == 4
    assert result["hard_forbidden"] is True
    assert result["constraint_status"] == "forbidden"
    assert result["matched_rules"][0]["rule_id"] == "TEST-INT-001"


def test_unverified_regulatory_rule():
    result = assess_pair(
        A, B,
        bindings=[make_binding("regulatory", False)],
    )
    assert result["final_risk"] == 3
    assert result["hard_forbidden"] is False
    assert result["constraint_status"] == "unknown"


def test_synthetic_rule_disabled():
    result = assess_pair(
        A, B,
        bindings=[make_binding("synthetic", False)],
    )
    assert result["final_risk"] == 3
    assert result["constraint_status"] == "unknown"


def test_synthetic_rule_enabled():
    result = assess_pair(
        A, B,
        bindings=[make_binding("synthetic", False)],
        enable_synthetic=True,
    )
    assert result["final_risk"] == 4
    assert result["hard_forbidden"] is True
