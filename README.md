# Zero-Shot Learning Implementation

Zero-shot text classification experiments using label semantics and transformers.

## Features

- **TF-IDF Baseline**: CPU-friendly zero-shot classification
- **Transformers Support**: Hugging Face transformers-based classification
- **Label Semantics**: Semantic similarity between inputs and labels
- **Exhaustive Notebooks**: Executed notebooks with saved outputs

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `zero_shot_classifier.ipynb` | Transformers-based prototype | ✅ |
| `01_zero_shot_tfidf_label_semantics.ipynb` | TF-IDF baseline with label semantics | ✅ Executed |

## Usage

```python
from zero_shot import classify

# Zero-shot classification
labels = ["sports", "politics", "technology"]
result = classify("The new iPhone was released today", labels)
print(result)  # {'label': 'technology', 'score': 0.85}
```

## Architecture

```
Zero-Shot-Learning-Implementation/
├── zero_shot_classifier.ipynb
├── 01_zero_shot_tfidf_label_semantics.ipynb
├── src/
│   ├── zero_shot.py
│   └── evaluation.py
├── tests/
└── docs/
```

## License

MIT
