"""Segmentation strategies — the study's independent variable (plan §7.1).

Each strategy maps a surface word to a list of *pre-tokens* (segments). These
pre-tokens are later fed to the model's own subword tokenizer (alignment.py).

Strategies
----------
- ``native``       : no pre-segmentation; the raw string goes straight to the
                     model tokenizer (offset-based alignment). Control / current
                     practice. For SentencePiece models this may merge across
                     whitespace; for WordPiece it coincides with ``whitespace``.
- ``whitespace``   : split on spaces -> one pre-token per word. Upper-bound ref.
- ``morphological``: split each word into morphemes (Morfessor, unsupervised).
- ``char``         : split each word into characters. Lower-bound ref.

The ``native`` strategy carries ``requires_pretokenization = False`` so the
aligner knows to use the raw-string path instead of ``is_split_into_words``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from .utils import get_logger

logger = get_logger()


@runtime_checkable
class Segmenter(Protocol):
    name: str
    requires_pretokenization: bool

    def segment_word(self, word: str) -> list[str]:
        """Split a single surface word into >=1 non-empty segments."""
        ...


@dataclass
class WhitespaceSegmenter:
    name: str = "whitespace"
    requires_pretokenization: bool = True

    def segment_word(self, word: str) -> list[str]:
        return [word]


@dataclass
class CharSegmenter:
    name: str = "char"
    requires_pretokenization: bool = True

    def segment_word(self, word: str) -> list[str]:
        return list(word) if word else [word]


@dataclass
class NativeSegmenter:
    """Sentinel: alignment uses the raw-string / offset path, not these splits."""

    name: str = "native"
    requires_pretokenization: bool = False

    def segment_word(self, word: str) -> list[str]:
        return [word]


class MorfessorSegmenter:
    """Unsupervised morphological segmentation via a trained Morfessor model.

    The model is trained on the training-split word types only (no external
    data), which keeps the strategy fully reproducible from our pipeline.
    """

    name = "morphological"
    requires_pretokenization = True

    def __init__(self, model) -> None:  # morfessor.BaselineModel
        self._model = model
        self._cache: dict[str, list[str]] = {}

    @classmethod
    def train(
        cls, words: Iterable[str], seed: int = 42, corpusweight: float = 1.0
    ) -> "MorfessorSegmenter":
        try:
            import morfessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "morphological segmentation needs `pip install morfessor`"
            ) from exc

        counts = Counter(w for w in words if w)
        # Pass the word STRING as the compound so the atom representation matches
        # what viterbi_segment uses at inference. (Passing tuple(word) instead
        # trains a char-tuple lexicon that viterbi — which re-splits the string —
        # cannot match, silently collapsing every word back to a single morph.)
        train_data = [(count, word) for word, count in counts.items()]
        model = morfessor.BaselineModel(corpusweight=corpusweight)
        model.load_data(train_data)
        model.train_batch()
        n_multi = sum(1 for *_, p in model.get_segmentations() if len(p) > 1)
        logger.info(
            "Trained Morfessor on %d word types (%d split into >1 morph)",
            len(counts), n_multi,
        )
        return cls(model)

    def segment_word(self, word: str) -> list[str]:
        if not word:
            return [word]
        if word not in self._cache:
            segments, _ = self._model.viterbi_segment(word)
            self._cache[word] = [s for s in segments if s] or [word]
        return self._cache[word]


class ZemberekSegmenter:
    """Optional rule-based Turkish morphology via the Zemberek JAR (plan §7.1).

    Not enabled by default: Zemberek is JVM-based and adds a heavy dependency.
    To use it, install a JPype/py4j bridge or call the JAR, implement
    ``segment_word`` to return the surface morpheme sequence, and register the
    name ``"morphological-zemberek"``. Left as a documented extension point so
    the core pipeline stays pure-Python and reproducible.
    """

    name = "morphological-zemberek"
    requires_pretokenization = True

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "ZemberekSegmenter is an optional extension; use MorfessorSegmenter "
            "for the reproducible default. See README 'Morphology backends'."
        )

    def segment_word(self, word: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError


def build_segmenter(
    name: str,
    *,
    train_words: Iterable[str] | None = None,
    seed: int = 42,
) -> Segmenter:
    """Factory. For ``morphological`` you must pass ``train_words`` (train split)."""
    name = name.lower()
    if name in ("native", "native-subword", "subword"):
        return NativeSegmenter()
    if name in ("whitespace", "word"):
        return WhitespaceSegmenter()
    if name in ("char", "character"):
        return CharSegmenter()
    if name in ("morphological", "morpheme", "morfessor"):
        if train_words is None:
            raise ValueError("morphological segmenter requires train_words to fit on")
        return MorfessorSegmenter.train(train_words, seed=seed)
    if name in ("morphological-zemberek", "zemberek"):
        return ZemberekSegmenter()
    raise ValueError(f"unknown segmentation strategy: {name!r}")


ALL_STRATEGIES = ("native", "whitespace", "morphological", "char")
