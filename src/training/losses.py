"""손실: 클래스 가중 교차 엔트로피 + 반발 보조 손실."""
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from src.schema import NUM_CLASSES, RELATIONS, max_grade


def class_weights(train_graphs: Sequence, cap: float = 8.0) -> Dict[str, torch.Tensor]:
    """w_c = min(cap, sqrt(N_0 / N_c)). 학습셋 빈도로만 계산한다."""
    weights = {}
    for r in RELATIONS:
        counts = torch.zeros(NUM_CLASSES[r], dtype=torch.float64)
        for g in train_graphs:
            counts += torch.bincount(g.y[r], minlength=NUM_CLASSES[r]).double()
        ratio = torch.sqrt(counts[0] / counts.clamp_min(1.0))
        weights[r] = ratio.clamp(max=cap).float()
        weights[r][0] = 1.0
    return weights


def total_loss(output, labels, weights, lambda_rep: float) -> torch.Tensor:
    """관계 평균 가중 CE + lambda_rep * 반발 보조 손실.

    보조 손실은 면역 보정 전의 rho가 정규화된 실제 등급(0~1)을 따라가게 한다.
    """
    ce = sum(F.cross_entropy(output.logits[r], labels[r], weight=weights[r])
             for r in RELATIONS) / len(RELATIONS)
    if not output.base_repulsion or lambda_rep <= 0:
        return ce
    aux = sum(
        F.mse_loss(output.base_repulsion[r].squeeze(-1),
                   labels[r].float() / max_grade(r))
        for r in RELATIONS) / len(RELATIONS)
    return ce + lambda_rep * aux
