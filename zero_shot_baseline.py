from __future__ import annotations

from math import sqrt


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same size")
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def predict_label(embedding: list[float], label_embeddings: dict[str, list[float]]) -> str:
    scored = {
        label: cosine_similarity(embedding, vec)
        for label, vec in label_embeddings.items()
    }
    return max(scored, key=scored.get)
