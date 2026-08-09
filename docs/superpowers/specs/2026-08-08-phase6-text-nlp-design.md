# Phase 6 Design — Text + NLP → M5

Date: 2026-08-08 | Status: draft

## Scope

Add text data support (TextDataSource), a pluggable tokenizer interface,
and an M5 example notebook (IMDB sentiment with LSTM). Zero new framework
dependencies in core. torch + torchvision used in the M5 notebook only.

## Components

### 1. `pipeline/data/text_source.py` — TextDataSource

DataStream that reads text from CSV files. Matches CsvDataSource pattern:
lazy-load, batch_size, shuffle, seed.

```python
class TextDataSource(DataStream):
    """CSV text column -> Batch iterator.
    
    Reads a CSV where one column contains raw text and another contains
    integer labels. Yields Batch(inputs=list[str], targets=np.ndarray).
    """
    def __init__(self, file_path, text_column="text", label_column="label",
                 batch_size=32, shuffle=True, seed=42): ...
    
    @property
    def vocab(self) -> set[str]: ...  # unique words across all texts
    @property
    def n_samples(self) -> int: ...
```

Batch format: `inputs = list[str]` (one string per sample in batch),
`targets = np.ndarray` of int labels.

### 2. `pipeline/data/tokenizer.py` — Simple whitespace tokenizer

```python
class SimpleTokenizer:
    """Build vocabulary from texts, encode text -> list[int]."""
    
    def __init__(self, max_vocab: int = 10000, min_freq: int = 1): ...
    def fit(self, texts: list[str]) -> "SimpleTokenizer": ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
```

Works as a transform: `TransformedDataStream(text_source, tokenizer_transform)` converts `list[str]` → `np.ndarray` of token IDs.

### 3. M5 notebook — IMDB sentiment

LSTM model with TorchAdapter on IMDB dataset (downloaded via torchvision.datasets or torchtext). Architecture: Embedding → LSTM → Linear → Sigmoid. Target > 85% accuracy.

---

## Self-review

- [x] TextDataSource follows CsvDataSource pattern exactly
- [x] SimpleTokenizer is framework-agnostic (pure Python dict + list)
- [x] Tokenizer is a utility, not a pipeline stage — used via TransformedDataStream
- [x] M5 uses existing TorchAdapter — no new adapter needed
- [x] Zero new dependencies in pipeline/
- [x] Lazy pandas import (TextDataSource._load uses pd.read_csv)
