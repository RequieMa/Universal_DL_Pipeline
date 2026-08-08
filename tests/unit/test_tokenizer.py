"""Unit tests for pipeline.data.tokenizer -- SimpleTokenizer."""

from __future__ import annotations

import pytest

from pipeline.data.tokenizer import SimpleTokenizer


@pytest.fixture
def fitted() -> SimpleTokenizer:
    """A tokenizer fitted on a small corpus."""
    texts = ["the quick brown fox", "the lazy dog", "fox jumps over dog"]
    return SimpleTokenizer(max_vocab=100, min_freq=1).fit(texts)


class TestSimpleTokenizerFit:
    """Vocabulary building tests."""

    def test_fit_vocab_size(self, fitted):
        """Happy Path: <PAD> and <UNK> plus unique words."""
        # tokens: the, quick, brown, fox, lazy, dog, jumps, over = 8
        # plus 2 special tokens = 10
        assert fitted.vocab_size == 10
        assert fitted.is_fitted

    def test_pad_token_is_zero(self, fitted):
        """Happy Path: <PAD> is always id 0 and <UNK> is id 1."""
        assert fitted.encode("<PAD>") == [0]
        assert fitted.encode("<UNK>") == [1]

    def test_fit_returns_self(self):
        """Happy Path: fit() returns self for chaining."""
        t = SimpleTokenizer()
        assert t.fit(["hello world"]) is t

    def test_min_freq_filter(self):
        """Boundary: words below min_freq are dropped."""
        # "hello" appears once, "world" appears twice
        t = SimpleTokenizer(max_vocab=100, min_freq=2).fit(["hello world", "world again"])
        assert "world" in t._word2idx
        assert "hello" not in t._word2idx
        assert "again" not in t._word2idx

    def test_max_vocab_truncation(self):
        """Boundary: max_vocab truncates to the most frequent words."""
        # 9 distinct words; max_vocab=4 keeps 2 special + 2 most frequent
        texts = ["a", "a", "a", "b", "b", "c"]
        t = SimpleTokenizer(max_vocab=4, min_freq=1).fit(texts)
        assert t.vocab_size == 4
        assert "a" in t._word2idx
        assert "b" in t._word2idx
        assert "c" not in t._word2idx


class TestSimpleTokenizerEncodeDecode:
    """Encoding and decoding tests."""

    def test_encode_decode_roundtrip(self, fitted):
        """Happy Path: in-vocab text encodes then decodes back to original."""
        text = "the quick brown fox"
        ids = fitted.encode(text)
        assert fitted.decode(ids) == text

    def test_oov_maps_to_unk(self, fitted):
        """Boundary: out-of-vocabulary words map to <UNK> id 1."""
        ids = fitted.encode("zzz_nonexistent_word")
        assert ids == [1]

    def test_empty_text_encodes_to_empty(self, fitted):
        """Boundary: empty text encodes to an empty token list."""
        assert fitted.encode("") == []
        assert fitted.decode([]) == ""

    def test_decode_unknown_id_leaks_unk(self, fitted):
        """Boundary: an id not present in the vocab decodes to <UNK>."""
        assert fitted.decode([99999]) == "<UNK>"


class TestSimpleTokenizerState:
    """State and error handling tests."""

    def test_unfitted_raises(self):
        """Error: encode/decode before fit raises RuntimeError."""
        t = SimpleTokenizer()
        assert not t.is_fitted
        assert t.vocab_size == 0
        with pytest.raises(RuntimeError):
            t.encode("hello")
        with pytest.raises(RuntimeError):
            t.decode([0])

    def test_max_vocab_too_small_raises(self):
        """Error: max_vocab < 2 cannot hold both special tokens."""
        t = SimpleTokenizer(max_vocab=1)
        with pytest.raises(ValueError):
            t.fit(["hello world"])

    def test_empty_corpus_still_has_specials(self):
        """Boundary: fitting on an empty corpus keeps <PAD> and <UNK>."""
        t = SimpleTokenizer().fit([])
        assert t.vocab_size == 2
