"""Simple text tokenizer.

Provides :class:`SimpleTokenizer`, a pure-Python vocabulary builder and
token encoder for teaching purposes. Whitespace tokenization with
``<PAD>`` and ``<UNK>`` special tokens.
"""

from __future__ import annotations

from collections import Counter


class SimpleTokenizer:
    """Frequency-based vocabulary builder and encoder.

    Splits text on whitespace, builds a frequency-ranked vocabulary with
    ``<PAD>`` and ``<UNK>`` as the first two tokens, and maps words to
    integer ids. Call :meth:`fit` once with a corpus before encoding.

    Usage::

        tokenizer = SimpleTokenizer(max_vocab=1000)
        tokenizer.fit(["hello world", "hello again"])
        ids = tokenizer.encode("hello world")  # -> [2, 3]
    """

    def __init__(self, max_vocab: int = 10000, min_freq: int = 1) -> None:
        """Create a tokenizer.

        Args:
            max_vocab: Maximum number of tokens, including ``<PAD>`` and
                ``<UNK>``. Truncation keeps the most frequent words.
            min_freq: Minimum corpus frequency for a word to be kept.
        """
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self._word2idx: dict[str, int] = {}
        self._idx2word: dict[int, str] = {}
        self._fitted = False

    def fit(self, texts: list[str]) -> SimpleTokenizer:
        """Build the vocabulary from a corpus.

        Args:
            texts: An iterable of raw strings.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``max_vocab`` is too small to hold both special
                tokens (``<PAD>`` and ``<UNK>``).
        """
        if self.max_vocab < 2:
            raise ValueError("max_vocab must be at least 2 for <PAD> and <UNK>")

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(str(text).split())

        self._word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, count in counter.most_common(self.max_vocab - 2):
            if count < self.min_freq:
                break
            idx = len(self._word2idx)
            self._word2idx[word] = idx
        self._idx2word = {v: k for k, v in self._word2idx.items()}
        self._fitted = True
        return self

    def encode(self, text: str) -> list[int]:
        """Map a raw string to a list of integer token ids.

        Args:
            text: The raw text to encode.

        Returns:
            A list of integer ids, one per whitespace token. Out-of-
            vocabulary words map to the ``<UNK>`` id.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._require_fitted()
        unk = self._word2idx["<UNK>"]
        return [self._word2idx.get(w, unk) for w in str(text).split()]

    def decode(self, tokens: list[int]) -> str:
        """Map a list of integer token ids back to a string.

        Args:
            tokens: Integer token ids.

        Returns:
            The decoded string; unknown ids map to ``<UNK>``.
        """
        self._require_fitted()
        return " ".join(self._idx2word.get(t, "<UNK>") for t in tokens)

    def _require_fitted(self) -> None:
        """Raise if the tokenizer has not been fitted."""
        if not self._fitted:
            raise RuntimeError("SimpleTokenizer must be fit() before encoding/decoding")

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary, including special tokens."""
        return len(self._word2idx)

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted
