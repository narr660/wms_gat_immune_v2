"""지표. 시퀀스별 혼동행렬을 저장해 두고 모든 지표를 그 위에서 계산한다.

혼동행렬 배열: 관계별 [시퀀스 수, K, K], 행 = 정답, 열 = 예측.
"""
from typing import Dict, Optional

import numpy as np
import torch

from src.schema import NUM_CLASSES, RELATIONS


@torch.no_grad()
def collect_confusion(model, graphs, device, memories=None):
    """memories: 그래프별 {relation: EdgeMemory} 리스트. 주면 aiNet 기억을 주입한다.

    반환: (시퀀스별 혼동행렬, 반발강도 오차)
    반발강도 오차 = |rho - 실제 등급/최고 등급| 의 평균. rho 는 PART 3에서 위험
    비용으로 쓰이므로, 위험 추정이 얼마나 정확한지를 따로 본다.
    """
    from src.immune.ainet import to_device
    from src.schema import max_grade
    model.eval()
    cms = {r: np.zeros((len(graphs), NUM_CLASSES[r], NUM_CLASSES[r]), dtype=np.int64)
           for r in RELATIONS}
    error = {r: [] for r in RELATIONS}
    for i, g in enumerate(graphs):
        memory = to_device(memories[i], device) if memories is not None else None
        out = model(g.x.to(device), g.edge_index.to(device), memory=memory)
        for r in RELATIONS:
            k = NUM_CLASSES[r]
            pred = out.logits[r].argmax(-1).cpu()
            cms[r][i] = torch.bincount(g.y[r] * k + pred, minlength=k * k).view(k, k).numpy()
            if r in out.repulsion:
                target = g.y[r].float() / max_grade(r)
                error[r].append(float((out.repulsion[r].squeeze(-1).cpu() - target).abs().mean()))
    rho_error = {r: float(np.mean(v)) for r, v in error.items() if v}
    if rho_error:
        rho_error["mean"] = float(np.mean(list(rho_error.values())))
    return cms, rho_error


def _prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    true_n, pred_n = cm.sum(1).astype(float), cm.sum(0).astype(float)
    precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    recall = np.divide(tp, true_n, out=np.zeros_like(tp), where=true_n > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1, true_n


def relation_metrics(cm: np.ndarray) -> Dict[str, float]:
    """한 관계의 합산 혼동행렬 [K, K] -> 지표."""
    precision, recall, f1, true_n = _prf(cm)
    present = true_n > 0                      # 실제로 존재하는 등급만 평균
    k = cm.shape[0]
    mid = np.arange(1, k - 1)
    mid = mid[present[mid]]
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()),
        "low_f1": float(f1[0]),
        "mid_f1": float(f1[mid].mean()) if len(mid) else float("nan"),
        "high_f1": float(f1[-1]),
        "high_recall": float(recall[-1]),
        "high_precision": float(precision[-1]),
        "high_total": int(cm[-1].sum()),
        "high_missed": int(cm[-1].sum() - cm[-1, -1]),      # 고위험 미탐지 수
        "low_total": int(cm[0].sum()),
        "low_overestimated": int(cm[0].sum() - cm[0, 0]),   # 저위험 과대평가 수
    }


_MEAN_KEYS = ("accuracy", "macro_f1", "low_f1", "mid_f1", "high_f1",
              "high_recall", "high_precision")
_SUM_KEYS = ("high_total", "high_missed", "low_total", "low_overestimated")


def summarize(cms: Dict[str, np.ndarray], indices: Optional[np.ndarray] = None) -> Dict:
    """시퀀스별 혼동행렬 -> 관계별 지표와 관계 평균 지표."""
    per_relation = {}
    for r in RELATIONS:
        selected = cms[r] if indices is None else cms[r][indices]
        per_relation[r] = relation_metrics(selected.sum(0))
    overall = {k: float(np.nanmean([per_relation[r][k] for r in RELATIONS]))
               for k in _MEAN_KEYS}
    overall.update({k: int(sum(per_relation[r][k] for r in RELATIONS)) for k in _SUM_KEYS})
    return {"overall": overall, "relations": per_relation}
