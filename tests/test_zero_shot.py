"""
Tests for Zero-Shot Learning Implementation.
"""

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def cosine_similarity(a, b):
    """Calculate cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def tfidf_label_similarity(text, labels):
    """Simple TF-IDF based zero-shot classification."""
    vectorizer = TfidfVectorizer()
    
    # Fit on all texts
    all_texts = [text] + labels
    vectorizer.fit(all_texts)
    
    # Transform
    text_vec = vectorizer.transform([text]).toarray()[0]
    label_vecs = vectorizer.transform(labels).toarray()
    
    # Calculate similarities
    similarities = []
    for label_vec in label_vecs:
        sim = cosine_similarity(text_vec, label_vec)
        similarities.append(sim)
    
    # Return best label
    best_idx = np.argmax(similarities)
    return {
        "label": labels[best_idx],
        "score": similarities[best_idx],
        "all_scores": dict(zip(labels, similarities))
    }


class TestZeroShot:
    """Test cases for zero-shot classification."""
    
    def test_similarity_returns_dict(self):
        """Test that classification returns a dictionary."""
        result = tfidf_label_similarity(
            "The football match was exciting",
            ["sports", "politics", "technology"]
        )
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result
    
    def test_classification_correct(self):
        """Test classification returns expected label."""
        result = tfidf_label_similarity(
            "The government passed a new law",
            ["sports", "politics", "technology"]
        )
        assert result["label"] == "politics"
    
    def test_all_scores_returned(self):
        """Test all scores are returned."""
        labels = ["sports", "politics", "technology"]
        result = tfidf_label_similarity("Breaking news report", labels)
        assert len(result["all_scores"]) == 3
    
    def test_different_inputs(self):
        """Test with various inputs."""
        test_cases = [
            ("Python is great for AI", ["sports", "technology", "health"]),
            ("The team won the championship", ["politics", "sports", "entertainment"]),
        ]
        
        for text, labels in test_cases:
            result = tfidf_label_similarity(text, labels)
            assert result["label"] in labels
            assert 0.0 <= result["score"] <= 1.0


class TestCosineSimilarity:
    """Test cosine similarity calculation."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors is 1."""
        vec = np.array([1, 0, 0])
        sim = cosine_similarity(vec, vec)
        assert sim == 1.0
    
    def test_perpendicular_vectors(self):
        """Test similarity of perpendicular vectors is 0."""
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1])
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
