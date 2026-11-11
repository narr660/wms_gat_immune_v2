"""벡터화한 레이블 계산이 원래 규칙(synthetic_oracle)과 같은지 검증한다."""
from src.data.graphs import complete_edges
from src.data.labels import pair_labels
from src.data.oracle import relation_risks
from src.schema import NUM_CLASSES, RELATIONS


def test_vectorized_labels_match_oracle(sequences):
    for sequence in sequences[:6]:
        edges = complete_edges(len(sequence))
        labels = pair_labels(sequence, edges[0], edges[1])
        for e, (i, j) in enumerate(edges.T):
            expected = relation_risks(sequence[i], sequence[j])
            for r in RELATIONS:
                assert labels[r][e] == expected[r], (r, i, j)


def test_labels_stay_within_class_range(sequences):
    for sequence in sequences:
        edges = complete_edges(len(sequence))
        labels = pair_labels(sequence, edges[0], edges[1])
        for r in RELATIONS:
            assert labels[r].min() >= 0 and labels[r].max() < NUM_CLASSES[r]
