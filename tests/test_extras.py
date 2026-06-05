"""Tests for evaluate/model extras (need torch+seqeval; run via pytest locally).

    python -m pytest tests/test_extras.py -q
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluate import repair_bio


def test_repair_bio_fixes_dangling_i():
    # I-X with no preceding B-X/I-X must become B-X
    assert repair_bio(["O", "I-time", "I-time", "O", "I-date"]) == [
        "O", "B-time", "I-time", "O", "B-date"
    ]


def test_repair_bio_fixes_type_switch():
    # I-city right after loc-span is invalid -> B-city
    assert repair_bio(["B-loc", "I-loc", "I-city"]) == ["B-loc", "I-loc", "B-city"]


def test_repair_bio_leaves_valid_unchanged():
    valid = ["O", "B-time", "I-time", "O", "B-date", "I-date"]
    assert repair_bio(valid) == valid


# ── CRF correctness vs brute force ───────────────────────────────────────────
def _brute_logZ_and_best(crf, em, length):
    """Enumerate all tag paths of given length; return (logZ, best_path)."""
    import itertools
    import math

    C = crf.num_tags
    start = crf.start.detach(); end = crf.end.detach(); trans = crf.trans.detach()
    scores = {}
    for path in itertools.product(range(C), repeat=length):
        s = float(start[path[0]] + em[0, path[0]])
        for t in range(1, length):
            s += float(trans[path[t - 1], path[t]] + em[t, path[t]])
        s += float(end[path[-1]])
        scores[path] = s
    logZ = math.log(sum(math.exp(s) for s in scores.values()))
    best = max(scores, key=scores.get)
    return logZ, list(best)


def test_crf_partition_and_viterbi_match_brute_force():
    import torch

    from src.model import LinearChainCRF

    torch.manual_seed(0)
    C, T = 3, 4
    crf = LinearChainCRF(C)
    with torch.no_grad():  # random but spread-out params
        crf.start.uniform_(-1, 1); crf.end.uniform_(-1, 1); crf.trans.uniform_(-1, 1)
    em = torch.randn(T, C)
    emis = em.unsqueeze(1)  # (T,B=1,C)
    mask = torch.ones(T, 1, dtype=torch.bool)

    logZ_brute, best_brute = _brute_logZ_and_best(crf, em, T)
    logZ = float(crf._denominator(emis, mask)[0].detach())
    assert abs(logZ - logZ_brute) < 1e-4, (logZ, logZ_brute)
    assert crf.decode(emis, mask)[0] == best_brute

    # numerator of a specific path equals that path's brute score
    path = [2, 0, 1, 2]
    tags = torch.tensor(path).unsqueeze(1)
    st, tr, en = crf.start.detach(), crf.trans.detach(), crf.end.detach()
    s = float(st[path[0]] + em[0, path[0]])
    for t in range(1, T):
        s += float(tr[path[t - 1], path[t]] + em[t, path[t]])
    s += float(en[path[-1]])
    assert abs(float(crf._numerator(emis, tags, mask)[0].detach()) - s) < 1e-4


def test_crf_handles_padding():
    import torch

    from src.model import LinearChainCRF

    torch.manual_seed(1)
    C, T, L = 3, 5, 3  # sequence of length 3, padded to 5
    crf = LinearChainCRF(C)
    with torch.no_grad():
        crf.start.uniform_(-1, 1); crf.end.uniform_(-1, 1); crf.trans.uniform_(-1, 1)
    em = torch.randn(T, C)
    emis = em.unsqueeze(1)
    mask = torch.tensor([[1], [1], [1], [0], [0]], dtype=torch.bool)
    logZ_brute, best_brute = _brute_logZ_and_best(crf, em[:L], L)
    assert abs(float(crf._denominator(emis, mask)[0].detach()) - logZ_brute) < 1e-4
    assert crf.decode(emis, mask)[0] == best_brute
