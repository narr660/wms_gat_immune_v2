
from typing import Dict


def odor_risk(a: Dict, b: Dict) -> int:
    """A에서 B로의 합성 냄새 전이 위험."""
    if a["odor_profile"] != "release":
        return 0
    if b["odor_profile"] != "absorb":
        return 0

    if a["sealed"] and b["sealed"]:
        return 1
    if a["sealed"] or b["sealed"]:
        return 2
    return 3


def contamination_risk(a: Dict, b: Dict) -> int:
    """A가 B를 오염시킬 잠재적 위험."""
    hazard = a["hazard_type"]
    leak = float(a["leak_risk"])
    level = float(a["hazard_level"])

    # 누출원 노출도
    exposure = leak * (0.4 if a["sealed"] else 1.0)

    # 수신 상품의 합성 민감도
    sensitivity = 1.0
    if b["category"] == "식품":
        sensitivity += 0.5
    if b["is_electronic"]:
        sensitivity += 0.3
    if not b["sealed"]:
        sensitivity += 0.2

    source_factor = 1.0 if hazard != "none" else 0.5
    score = exposure * sensitivity * source_factor

    if hazard in ("toxic", "corrosive") and level >= 0.7 and score >= 0.8:
        return 4
    if score >= 0.65:
        return 3
    if score >= 0.35:
        return 2
    if score >= 0.15:
        return 1
    return 0


def physical_risk(a: Dict, b: Dict) -> int:
    """
    A와 B의 배치 전 잠재적 물리 손상 위험.
    실제 압축 하중 초과 여부는 별도 평가.
    """
    weight = float(a["weight_sim"])
    support = max(float(b["max_supported_weight_sim"]), 1e-6)

    ratio = weight / support

    if b["fragile"]:
        if ratio >= 1.5:
            return 4
        if ratio >= 1.0:
            return 3
        if ratio >= 0.5:
            return 2
        return 1

    if ratio >= 2.0:
        return 3
    if ratio >= 1.0:
        return 2
    if ratio >= 0.5:
        return 1
    return 0


def relation_risks(a: Dict, b: Dict) -> Dict[str, int]:
    return {
        "odor": odor_risk(a, b),
        "contamination": contamination_risk(a, b),
        "physical": physical_risk(a, b),
    }
