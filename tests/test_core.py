"""No-network unit checks for the parsing / projection / segmentation core.

Run:  python tests/test_core.py   (exits 0 on success)
These do not need a model or internet — only numpy + morfessor.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile

from src.alignment import project_tag, segment_with_tags
from src.data import build_label_maps, load_atis_format, load_examples, parse_annot_utt
from src.segmentation import CharSegmenter, WhitespaceSegmenter, build_segmenter


def test_parse_annot_utt():
    toks, tags = parse_annot_utt("[date : yarın] sabah [time : yedi] için alarm kur")
    assert toks == ["yarın", "sabah", "yedi", "için", "alarm", "kur"], toks
    assert tags == ["B-date", "O", "B-time", "O", "O", "O"], tags

    # multi-token slot value -> B then I
    toks, tags = parse_annot_utt("[date : her gün] [time : altıda] beni uyandır")
    assert toks == ["her", "gün", "altıda", "beni", "uyandır"], toks
    assert tags == ["B-date", "I-date", "B-time", "O", "O"], tags


def test_project_tag():
    assert project_tag("O", 3) == ["O", "O", "O"]
    assert project_tag("B-time", 3) == ["B-time", "I-time", "I-time"]
    assert project_tag("I-time", 2) == ["I-time", "I-time"]
    assert project_tag("B-loc", 1) == ["B-loc"]


def test_segment_with_tags():
    words = ["faturalarımdan", "Ankara"]
    tags = ["B-amount", "B-place"]
    pre, ptags, widx = segment_with_tags(words, tags, CharSegmenter())
    assert len(pre) == len("faturalarımdan") + len("Ankara")
    assert ptags[0] == "B-amount" and ptags[1] == "I-amount"  # first char B, rest I
    assert widx[0] == 0 and widx[-1] == 1

    pre, ptags, widx = segment_with_tags(words, tags, WhitespaceSegmenter())
    assert pre == words and ptags == tags and widx == [0, 1]


def test_label_maps():
    ex = load_examples(pathlib.Path(__file__).with_name("sample_tr.jsonl"), "train")
    lm = build_label_maps(ex)
    assert lm.slot2id["O"] == 0
    assert lm.num_intents >= 3
    assert all(t.startswith(("B-", "I-")) for t in lm.slot2id if t != "O")


def test_atis_loader():
    with tempfile.TemporaryDirectory() as d:
        dd = pathlib.Path(d)
        (dd / "seq.in").write_text("ankaraya uçuş ara\nyarın istanbul\n", encoding="utf-8")
        (dd / "seq.out").write_text("B-toloc O O\nB-date B-toloc\n", encoding="utf-8")
        (dd / "label").write_text("flight\nflight\n", encoding="utf-8")
        exs = load_atis_format(dd)
    assert len(exs) == 2
    assert exs[0].tokens == ["ankaraya", "uçuş", "ara"]
    assert exs[0].slots == ["B-toloc", "O", "O"]
    assert exs[0].intent == "flight" and exs[0].scenario == "atis"


def test_morfessor_optional():
    try:
        import morfessor  # noqa: F401
    except ImportError:
        print("  (morfessor not installed — skipping morphological check)")
        return
    seg = build_segmenter("morphological", train_words=["evlerimden", "evler", "evden", "ev"])
    parts = seg.segment_word("evlerimden")
    assert isinstance(parts, list) and "".join(parts) == "evlerimden", parts
    print(f"  morfessor split: evlerimden -> {parts}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll core tests passed.")
