import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from app.evaluation import retrieval_metrics as m


def test_recall_hand_calculated():
    # ranked grades [1,0,1,0,0]; 4 relevant in corpus
    assert m.recall_at_k([1, 0, 1, 0, 0], 4, 3) == pytest.approx(2 / 4)
    assert m.recall_at_k([1, 0, 1, 0, 0], 4, 1) == pytest.approx(1 / 4)
    assert m.recall_at_k([0, 0], 0, 2) is None


def test_hit_rate_vs_recall_differ():
    assert m.hit_at_k([0, 0, 1], 3) == 1.0 and m.hit_at_k([0, 0, 1], 2) == 0.0
    assert m.recall_at_k([0, 0, 1], 2, 3) == 0.5


def test_mrr():
    assert m.reciprocal_rank([0, 0, 1], 5) == pytest.approx(1 / 3)
    assert m.reciprocal_rank([0, 0, 1], 2) == 0.0       # cut-off respected
    assert m.reciprocal_rank([], 3) == 0.0
    assert m.mean([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_ndcg_hand_calculated():
    # DCG = 2/log2(3) + 1/log2(4) ; IDCG = 2/log2(2) + 1/log2(3)
    expected = (2 / math.log2(3) + 1 / math.log2(4)) / (2 / 1 + 1 / math.log2(3))
    assert m.ndcg_at_k([0, 2, 1], [2, 1], 3) == pytest.approx(expected)
    assert m.ndcg_at_k([2, 1], [2, 1], 2) == pytest.approx(1.0)
    assert m.ndcg_at_k([0, 0], [], 2) is None


def test_ndcg_cutoff_and_order():
    assert m.ndcg_at_k([0, 0, 2], [2], 2) == 0.0
    assert m.ndcg_at_k([1, 2], [2, 1], 2) < m.ndcg_at_k([2, 1], [2, 1], 2)


def test_ndcg_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        grades = rng.integers(0, 3, size=8)
        ideal_pool = list(rng.permutation(grades))
        # sklearn needs one score vector: use descending rank scores
        score = np.arange(len(grades), 0, -1)
        ours = m.ndcg_at_k(list(grades), list(grades), 5)
        if ours is None:
            continue
        assert ours == pytest.approx(ndcg_score([grades], [score], k=5))


def test_context_precision():
    assert m.context_precision_at_k([1, 0, 1, 0, 0], 5) == pytest.approx(2 / 5)
    assert m.context_precision_at_k([1, 0, 1], 2) == pytest.approx(1 / 2)
    assert m.context_precision_at_k([], 5) == 0.0      # documented empty behaviour


def test_evidence_recall_not_penalised_by_duplicate_chunks():
    assert m.evidence_recall([True, False]) == 0.5 and m.evidence_recall([]) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        m.hit_at_k([1], 0)
