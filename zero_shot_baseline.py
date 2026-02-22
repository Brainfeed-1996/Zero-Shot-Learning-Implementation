from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

class ZeroShotClassifier(ABC):
    @abstractmethod
    def predict(self, input_data: any) -> str:
        pass

class IndustrialZeroShotEngine(ZeroShotClassifier):
    """
    Industrial-grade Zero-Shot Engine with batch processing,
    robust error handling, and confidence scoring.
    """
    def __init__(self, label_embeddings: Dict[str, np.ndarray]):
        self.label_embeddings = label_embeddings
        self._validate_embeddings()

    def _validate_embeddings(self):
        if not self.label_embeddings:
            raise ValueError("Label embeddings cannot be empty")
        # Ensure all embeddings have same dimensionality
        dims = {v.shape for v in self.label_embeddings.values()}
        if len(dims) > 1:
            raise ValueError(f"Inconsistent embedding dimensions: {dims}")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def predict(self, embedding: np.ndarray) -> str:
        """Predict single label with industrial error handling."""
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        scores = {
            label: self.cosine_similarity(embedding, label_vec)
            for label, label_vec in self.label_embeddings.items()
        }
        return max(scores, key=scores.get)

    def predict_with_confidence(self, embedding: np.ndarray) -> Dict[str, any]:
        """Enriched prediction with confidence metrics for industrial monitoring."""
        scores = {
            label: float(self.cosine_similarity(embedding, label_vec))
            for label, label_vec in self.label_embeddings.items()
        }
        top_label = max(scores, key=scores.get)
        confidence = scores[top_label]
        
        # Entropy-based uncertainty (simplified)
        margin = confidence - sorted(scores.values())[-2] if len(scores) > 1 else 1.0
        
        return {
            "label": top_label,
            "confidence": confidence,
            "margin": margin,
            "all_scores": scores
        }

    def batch_predict(self, embeddings: List[np.ndarray]) -> List[str]:
        """Vectorized batch prediction for high-throughput pipelines."""
        return [self.predict(e) for e in embeddings]
