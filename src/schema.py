"""공통 정의: 위험 관계, 등급 수, 등급 그룹."""
from typing import Dict, Tuple

RELATIONS: Tuple[str, ...] = ("odor", "contamination", "physical")

# 레이블 규칙(synthetic_oracle)이 실제로 내는 등급 범위와 일치시킨다.
NUM_CLASSES: Dict[str, int] = {"odor": 4, "contamination": 5, "physical": 5}


def max_grade(relation: str) -> int:
    return NUM_CLASSES[relation] - 1


def grade_group(relation: str, grade: int) -> str:
    """Low = 0등급, High = 최고 등급, 그 사이는 Mid."""
    if grade == 0:
        return "low"
    if grade == max_grade(relation):
        return "high"
    return "mid"
