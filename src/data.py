"""MASSIVE-tr loading and the ``annot_utt`` -> word-level BIO parser.

We read the official Amazon MASSIVE release (a tar.gz of per-locale JSONL files)
rather than going through a HuggingFace loading script. This keeps the data path
deterministic and independent of `datasets` version churn (script loading was
deprecated/removed in recent releases). See plan §6.1.

The annotated utterance uses a bracket format:
    "saat [time : yedi] için [date : yarın] alarm kur"
Tokens inside ``[slot : ...]`` brackets receive B-/I- tags for that slot; all
other (whitespace) tokens are ``O``.
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests

from .utils import get_logger, read_jsonl

logger = get_logger()

MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.0.tar.gz"
)
DEFAULT_LOCALE = "tr-TR"
PARTITIONS = ("train", "dev", "test")


@dataclass
class Example:
    """One utterance: surface words + word-level BIO slot tags + intent."""

    uid: str
    tokens: list[str]
    slots: list[str]  # word-level BIO tags, len == len(tokens)
    intent: str
    scenario: str = ""

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.slots):
            raise ValueError(
                f"token/slot length mismatch in {self.uid}: "
                f"{len(self.tokens)} vs {len(self.slots)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# annot_utt parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_annot_utt(annot_utt: str) -> tuple[list[str], list[str]]:
    """Parse a MASSIVE ``annot_utt`` string into (tokens, BIO tags).

    State machine over whitespace tokens. ``[slot`` opens a slot, a lone ``:``
    separates name from value, and a token ending in ``]`` closes it.
    """
    tokens: list[str] = []
    tags: list[str] = []

    cur_slot: str | None = None
    expect_colon = False
    first_in_slot = False

    for raw in annot_utt.split():
        if raw.startswith("[") and cur_slot is None:
            cur_slot = raw[1:]
            expect_colon = True
            first_in_slot = True
            continue

        if expect_colon:
            expect_colon = False
            if raw == ":":
                continue  # canonical "[name : value]" separator
            # Fallback: separator was glued/missing -> treat `raw` as first value.

        closes = raw.endswith("]")
        word = raw[:-1] if closes else raw

        if cur_slot is not None:
            if word:
                tags.append(("B-" if first_in_slot else "I-") + cur_slot)
                tokens.append(word)
                first_in_slot = False
            if closes:
                cur_slot = None
        else:
            if word:
                tokens.append(word)
                tags.append("O")

    return tokens, tags


def example_from_record(rec: dict) -> Example:
    tokens, tags = parse_annot_utt(rec["annot_utt"])
    return Example(
        uid=str(rec.get("id", "")),
        tokens=tokens,
        slots=tags,
        intent=rec["intent"],
        scenario=rec.get("scenario", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Download / load
# ─────────────────────────────────────────────────────────────────────────────
def download_massive(raw_dir: str | Path, locale: str = DEFAULT_LOCALE) -> Path:
    """Download the official MASSIVE tar.gz and extract one locale's JSONL.

    Returns the path to ``<raw_dir>/<locale>.jsonl``. Idempotent: skips work if
    the locale file already exists.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{locale}.jsonl"
    if out.exists():
        logger.info("MASSIVE %s already present at %s", locale, out)
        return out

    archive = raw_dir / "amazon-massive-dataset-1.0.tar.gz"
    if not archive.exists():
        logger.info("Downloading MASSIVE (~330 MB) -> %s", archive)
        with requests.get(MASSIVE_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with archive.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

    member_name = f"1.0/data/{locale}.jsonl"
    logger.info("Extracting %s", member_name)
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        with tar.extractfile(member) as src, out.open("wb") as dst:  # type: ignore[union-attr]
            dst.write(src.read())
    logger.info("Wrote %s", out)
    return out


def load_examples(jsonl_path: str | Path, partition: str) -> list[Example]:
    """Load one split (train/dev/test) from a MASSIVE locale JSONL file."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    examples = [
        example_from_record(rec)
        for rec in read_jsonl(jsonl_path)
        if rec.get("partition") == partition
    ]
    logger.info("Loaded %d %s examples from %s", len(examples), partition, jsonl_path)
    return examples


def load_atis_format(split_dir: str | Path) -> list[Example]:
    """Load a JointBERT-style ATIS / MultiATIS++ split directory (plan §6.2).

    Expects three parallel files in ``split_dir`` (one example per line):
      seq.in   space-separated tokens
      seq.out  space-separated BIO slot tags (len == tokens)
      label    intent string
    This is the de-facto format of MultiATIS++ releases; point it at the Turkish
    ``tr`` split's train/dev/test folders to add a second domain (flight).
    """
    d = Path(split_dir)
    tok_lines = (d / "seq.in").read_text(encoding="utf-8").splitlines()
    tag_lines = (d / "seq.out").read_text(encoding="utf-8").splitlines()
    intents = (d / "label").read_text(encoding="utf-8").splitlines()
    examples, skipped = [], 0
    for i, (toks, tags, intent) in enumerate(zip(tok_lines, tag_lines, intents)):
        tokens, slots = toks.split(), tags.split()
        if not tokens or len(tokens) != len(slots):
            skipped += 1
            continue
        examples.append(
            Example(uid=str(i), tokens=tokens, slots=slots,
                    intent=intent.strip(), scenario="atis")
        )
    logger.info("Loaded %d examples from %s (skipped %d)", len(examples), d, skipped)
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LabelMaps:
    intent2id: dict[str, int]
    slot2id: dict[str, int]  # BIO labels incl. "O"
    id2intent: dict[int, str] = field(init=False)
    id2slot: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slots(self) -> int:
        return len(self.slot2id)


def build_label_maps(*splits: Iterable[Example]) -> LabelMaps:
    """Build intent/slot id maps from the training (and optionally dev) splits.

    "O" is pinned to id 0 for the slot map so it is a stable, conventional index.
    """
    intents: set[str] = set()
    slots: set[str] = set()
    for split in splits:
        for ex in split:
            intents.add(ex.intent)
            slots.update(ex.slots)
    slots.discard("O")
    intent2id = {name: i for i, name in enumerate(sorted(intents))}
    slot2id = {"O": 0, **{name: i + 1 for i, name in enumerate(sorted(slots))}}
    return LabelMaps(intent2id=intent2id, slot2id=slot2id)
