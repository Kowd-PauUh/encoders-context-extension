import math


def ndcg(
    top_hits: list,
    qrels: list,
    k_values: list[int] = [10],
):
    def _dcg_at_k(relevances, k):
        return sum(
            relevances[i] / math.log(i + 2, 2)  # +2 as we start indexing from 0
            for i in range(min(len(relevances), k))
        )

    if any(not isinstance(k, int) for k in k_values):
        raise TypeError('Each k value in `k_values` must be of type `int`.')
    if any(k <= 0 for k in k_values):
        raise ValueError('Each k value in `k_values` must be greater than 0.')

    ndcg_at_k = {}
    for k in k_values:
        predicted_relevances = [
            1 if hit in qrels else 0
            for hit in top_hits[:k]
        ]
        true_relevances = [1] * min(len(qrels), k)

        ideal_dcg = _dcg_at_k(true_relevances, k)
        ndcg_value = _dcg_at_k(predicted_relevances, k) / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_at_k[k] = ndcg_value

    return ndcg_at_k
