
from typing import Dict, List, Optional

from src.data.oracle import relation_risks
from src.constraints.hard_constraints import (
    RuleBinding,
    evaluate_hard_constraints,
)

RISK_NAMES = {
    0: "가능",
    1: "낮음",
    2: "중간",
    3: "높음",
    4: "직접 금지",
}


def assess_pair(
    a: Dict,
    b: Dict,
    bindings: Optional[List[RuleBinding]] = None,
    enable_synthetic: bool = False,
) -> Dict:
    """A가 B에 미치는 방향성 위험 및 금지 규칙 평가."""

    risks = relation_risks(a, b)

    # V4의 4등급은 잠재적 심각도로만 해석
    indirect_risk = min(3, max(risks.values()))

    constraint = evaluate_hard_constraints(
        a,
        b,
        bindings=bindings,
        enable_synthetic=enable_synthetic,
    )

    hard_forbidden = constraint["hard_forbidden"]

    final_risk = 4 if hard_forbidden else indirect_risk

    return {
        "relation_risks": risks,
        "indirect_risk": indirect_risk,
        "hard_forbidden": hard_forbidden,
        "constraint_status": constraint["evaluation_status"],
        "matched_rules": constraint["matched_rules"],
        "final_risk": final_risk,
        "risk_name": RISK_NAMES[final_risk],
    }
