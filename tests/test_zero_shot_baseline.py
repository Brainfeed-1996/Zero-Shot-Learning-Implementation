from zero_shot_baseline import cosine_similarity, predict_label


def test_cosine_similarity_identity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0


def test_predict_label():
    pred = predict_label([0.9, 0.1], {"sports": [1, 0], "finance": [0, 1]})
    assert pred == "sports"
