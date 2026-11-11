from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class EdgeMemory:
    """aiNet이 간선별로 주는 값. 모두 [E, 1]."""
    strength: torch.Tensor      # M: 위험 기억 강도 [0, 1]  (강화에 사용)
    direction: torch.Tensor     # s: 보정 방향 [-1, 1]
    confidence: torch.Tensor    # C: 관측 신뢰도 [0, 1]     (완화에 사용)
    target: Optional[torch.Tensor] = None   # 기억된 평균 관측 노출 [0, 1]. 보정은 이 값을
                                            # 향해 끌어당기되 지나치지 않는다 (과도 보정 방지)


@dataclass
class ModelOutput:
    logits: Dict[str, torch.Tensor]                     # 관계별 [E, K_r]
    embeddings: Dict[str, torch.Tensor]                 # 관계별 [N, H]
    base_repulsion: Dict[str, torch.Tensor] = field(default_factory=dict)   # GAT가 만든 rho
    repulsion: Dict[str, torch.Tensor] = field(default_factory=dict)        # 메시지 전달에 쓰인 최종 rho
    danger: Dict[str, torch.Tensor] = field(default_factory=dict)
    safe: Dict[str, torch.Tensor] = field(default_factory=dict)
    dca_repulsion: Dict[str, torch.Tensor] = field(default_factory=dict)    # DCA 보정 후, 기억 보정 전
    memory_delta: Dict[str, torch.Tensor] = field(default_factory=dict)     # 기억 보정량
