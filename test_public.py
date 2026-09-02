"""Public tests: the same checks the grader runs, on public instances.

Run them either way::

    python test_public.py        # no pytest needed
    pytest test_public.py

Passing all of these does not guarantee full marks: the grader uses
hidden instances built from your own student id, and they go outside the
ranges used here on purpose.  But every check below mirrors one the
grader makes, so if you fail here, you will definitely fail the grader.
"""

import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from synthid import (
    BloomFilter,
    bloom_scored_positions,
    context_item,
    g_value,
    histogram,
    mean_score,
    sample_knockout,
    sample_layered,
    score_numerator,
    scored_positions,
    threshold,
)
from toolkit import TOKENS, Randomness

TOLERANCE = 1e-9


def close(a: float, b: float, tolerance: float = TOLERANCE) -> bool:
    return abs(a - b) <= tolerance


def vectors_close(
    got: Sequence[float], want: Sequence[float], tolerance: float = TOLERANCE
) -> bool:
    return len(got) == len(want) and all(
        close(x, y, tolerance) for x, y in zip(got, want, strict=True)
    )


# ==========================================================================
# Task A -- worked example 1
# ==========================================================================

P0 = [0.4, 0.3, 0.2, 0.1]
G = [[1, 0, 1, 0], [0, 0, 1, 1]]
AFTER_LAYER_1 = [0.56, 0.12, 0.28, 0.04]
AFTER_LAYER_2 = [0.3808, 0.0816, 0.4704, 0.0672]


def test_worked_example_layer_1():
    """One layer of the update, against the handout's four numbers."""
    got = sample_layered(P0, G, 1)
    assert vectors_close(got, AFTER_LAYER_1), got


def test_worked_example_layer_2():
    """Both layers.  C started third and ends highest."""
    got = sample_layered(P0, G, 2)
    assert vectors_close(got, AFTER_LAYER_2), got


def test_sum_stays_one_with_a_tolerance():
    """Compare with a tolerance -- the exact test fails on correct code."""
    total = sum(sample_layered(P0, G, 2))
    assert close(total, 1.0), total


def test_probabilities_are_never_negative():
    probs = sample_layered(P0, G, 2)
    assert all(value >= 0.0 for value in probs), probs


# ==========================================================================
# Task A -- degenerate cases the grader also uses
# ==========================================================================


def test_all_g_zero_changes_nothing():
    """With no token favoured, a layer must leave p alone."""
    p = [0.30, 0.25, 0.20, 0.13, 0.08, 0.04]
    got = sample_layered(p, [[0] * 6] * 4, 4)
    assert vectors_close(got, p), got


def test_all_g_one_changes_nothing():
    p = [0.30, 0.25, 0.20, 0.13, 0.08, 0.04]
    got = sample_layered(p, [[1] * 6] * 4, 4)
    assert vectors_close(got, p), got


def test_a_single_atom_is_left_alone():
    """No freedom in the distribution means nothing for the mark to use."""
    p = [0.0, 0.0, 1.0, 0.0]
    got = sample_layered(p, [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], 3)
    assert vectors_close(got, p), got


def test_zero_probability_tokens_stay_at_zero():
    p = [0.0, 0.5, 0.0, 0.5]
    got = sample_layered(p, [[1, 0, 1, 0], [0, 0, 1, 1]], 2)
    assert got[0] == 0.0 and got[2] == 0.0, got


def test_layer_order_matters():
    """g[0] is applied first.  Reversing the layers is a different answer."""
    p = [0.30, 0.25, 0.20, 0.13, 0.08, 0.04]
    layers = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
    forward = sample_layered(p, layers, 2)
    backward = sample_layered(p, layers[::-1], 2)
    assert not vectors_close(forward, backward, 1e-6), forward


# ==========================================================================
# Task A -- the two implementations must agree
# ==========================================================================


def test_knockout_matches_layered():
    """Draw many tokens and compare the proportions with your own layered.

    This is the differential test.  It needs no expected values: if your
    two implementations disagree, at least one of them is wrong.
    """
    draws = 20_000
    rng = Randomness("test_knockout_matches_layered")
    expected = sample_layered(P0, G, 2)
    counts = Counter(sample_knockout(P0, G, 2, rng) for _ in range(draws))
    tolerance = 3 / draws**0.5
    for token, want in enumerate(expected):
        got = counts[token] / draws
        assert close(got, want, tolerance), (token, got, want)


def test_knockout_only_returns_tokens_that_have_probability():
    rng = Randomness("test_knockout_only_returns")
    p = [0.0, 0.5, 0.0, 0.5]
    layers = [[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]
    drawn = {sample_knockout(p, layers, 3, rng) for _ in range(300)}
    assert drawn <= {1, 3}, drawn


def test_the_knockout_draws_from_the_stream_in_order():
    """Two runs given equal streams must return the same token.

    If this fails, something in your knockout is taking randomness from
    somewhere other than the stream it was handed; everything you
    generate will differ from one run to the next.
    """
    p = [0.1, 0.2, 0.3, 0.4]
    layers = [[1, 0, 1, 0], [0, 1, 1, 0]]
    first = sample_knockout(p, layers, 2, Randomness("same", 1))
    again = sample_knockout(p, layers, 2, Randomness("same", 1))
    assert first == again, (first, again)


def test_ties_in_the_knockout_go_to_the_left():
    """Every g-value equal, so every match is a tie.

    With all g-values the same, no candidate beats another, and the
    left one survives every round. So the token that comes back is the
    *first* of the 2**m drawn, which is the one a stream of a single
    repeated value gives.

    """
    p = [0.25, 0.25, 0.25, 0.25]
    layers = [[1, 1, 1, 1], [1, 1, 1, 1]]
    rng = Randomness("ties", 99)
    expected = sample_knockout(p, layers, 0, Randomness("ties", 99))
    assert sample_knockout(p, layers, 2, rng) == expected


# ==========================================================================
# Task B1 -- which positions get scored
# ==========================================================================

KEY = "0123456789abcdef"


def test_the_first_h_positions_are_never_scored():
    tokens = [f"w{i:03d}" for i in range(20)]
    for h in (2, 3, 5, 6):
        assert min(scored_positions(tokens, h)) == h


def test_everything_is_scored_when_nothing_repeats():
    tokens = [f"w{i:03d}" for i in range(30)]
    assert scored_positions(tokens, 4) == list(range(4, 30))


def test_a_repeated_paragraph_is_scored_once():
    """The same four tokens three times over: evidence counts once."""
    tokens = ["a", "b", "c", "d"] * 3
    assert scored_positions(tokens, 2) == [2, 3, 4, 5]


def test_a_repeat_that_starts_mid_window():
    """Position 6 is new -- its window mixes both copies.  Position 7 is not."""
    tokens = ["x", "a", "b", "c", "y", "a", "b", "c"]
    assert scored_positions(tokens, 2) == [2, 3, 4, 5, 6]


def test_the_same_token_after_a_different_window_is_not_a_duplicate():
    tokens = ["a", "b", "z", "c", "b", "z"]
    assert scored_positions(tokens, 2) == [2, 3, 4, 5]


def test_a_document_shorter_than_the_window_scores_nothing():
    assert scored_positions(["a", "b"], 3) == []
    assert scored_positions(["a", "b", "c"], 3) == []
    assert scored_positions([], 3) == []


def test_item_encoding():
    assert context_item(["the", "cat"]) == "the|cat"


# ==========================================================================
# Task B1 -- what those positions score
# ==========================================================================


def test_score_numerator_adds_up_the_g_values():
    tokens = ["a", "b", "c", "d", "a", "b", "c", "d"]
    h, m = 2, 5
    expected = 0
    for i in scored_positions(tokens, h):
        window = tokens[i - h : i]
        expected += sum(g_value(KEY, window, layer, tokens[i])
                        for layer in range(m))
    assert score_numerator(tokens, KEY, h, m) == expected


def test_mean_score_averages_over_the_scored_positions():
    tokens = [f"w{i:03d}" for i in range(40)]
    h, m = 4, 9
    k = score_numerator(tokens, KEY, h, m)
    positions = len(scored_positions(tokens, h))
    assert close(mean_score(tokens, KEY, h, m), k / (m * positions))


def test_mean_score_of_an_empty_document():
    assert mean_score([], KEY, 3, 9) == 0.0


def test_doubling_a_document_does_not_raise_its_numerator():
    tokens = ["a", "b", "c", "d", "e"] * 2
    assert (score_numerator(tokens * 2, KEY, 2, 7) ==
            score_numerator(tokens, KEY, 2, 7))


# ==========================================================================
# Task B2 -- worked example 2, and the Bloom filter guarantee
# ==========================================================================

# "the cat sat on the mat" at h = 2 gives the contexts below.
ITEM_A = "the|cat"
ITEM_B = "cat|sat"
ABSENT = "sat|on"
FALSE_POSITIVE = "mat|on"


def test_worked_example_2_queries():
    """The 16-bit toy from the handout, checked through add and contains."""
    filt = BloomFilter(16, 3)
    filt.add(ITEM_A)
    filt.add(ITEM_B)
    assert filt.contains(ITEM_A)
    assert filt.contains(ITEM_B)
    assert not filt.contains(ABSENT), "bit 4 is clear, so this must be absent"
    assert filt.contains(FALSE_POSITIVE), "all three bits are set: a false positive"


def test_no_false_negatives():
    """Everything added reports present. This is the Bloom guarantee."""
    for nbits in (512, 2048, 8192):
        filt = BloomFilter(nbits, 3)
        items = [
            context_item([f"w{i:03d}", f"w{i + 1:03d}", f"w{i + 2:03d}"])
            for i in range(600)
        ]
        for item in items:
            filt.add(item)
            assert filt.contains(item), (nbits, item)
        for item in items:
            assert filt.contains(item), (nbits, item)


def test_an_empty_filter_claims_nothing():
    filt = BloomFilter(512, 3)
    assert not filt.contains("anything|at>all")


def test_the_backing_store_is_small_enough():
    """The size check, run the way the grader runs it.

    Everything the filter object can reach is measured. A packed
    bytearray or a single int passes; a list of nbits integers or a
    set of the items does not.

    """
    for nbits in (512, 2048, 8192):
        filt = BloomFilter(nbits, 3)
        for i in range(400):
            filt.add(
                context_item([f"w{i:03d}", f"w{i + 1:03d}", f"w{i + 2:03d}"])
            )
        measured = reachable_bytes(filt)
        limit = nbits // 8 + 512
        assert measured <= limit, (
            f"{nbits}-bit filter reaches {measured} bytes, limit {limit}"
        )


def reachable_bytes(obj: object) -> int:
    """Total size of everything ``obj`` stores, counting each object once."""
    seen: set[int] = set()
    total = 0
    pending: list[object] = (
        list(vars(obj).values())
        if hasattr(obj, "__dict__")
        else []
    )
    for name in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, name):
            pending.append(getattr(obj, name))
    while pending:
        item = pending.pop()
        if id(item) in seen or callable(item):
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
    return total


def test_the_filter_only_ever_loses_positions():
    """A false positive skips a position; it can never invent one."""
    tokens = [f"w{(i * 7) % 1000:03d}" for i in range(400)]
    exact = set(scored_positions(tokens, 3))
    for nbits in (512, 2048, 8192):
        filtered = set(bloom_scored_positions(tokens, 3, nbits, 3))
        assert filtered <= exact, nbits


def test_more_bits_means_fewer_false_skips():
    tokens = [f"w{(i * 7) % 1000:03d}" for i in range(400)]
    exact = len(scored_positions(tokens, 3))
    skips = [
        exact - len(bloom_scored_positions(tokens, 3, nbits, 3))
        for nbits in (512, 2048, 8192)
    ]
    assert skips == sorted(skips, reverse=True), skips


# ==========================================================================
# Task C -- the counting sort and the threshold rule
# ==========================================================================


def test_histogram_has_one_slot_per_possible_score():
    assert len(histogram([], 200, 9)) == 9 * 200 + 1


def test_histogram_counts_each_document_once():
    numerators = [3, 3, 7, 0, 12]
    hist = histogram(numerators, 25, 9)
    assert sum(hist) == 5
    assert hist[3] == 2 and hist[7] == 1 and hist[0] == 1 and hist[12] == 1


def test_threshold_leaves_at_most_alpha_in_the_tail():
    hist = [0] * 101
    hist[50] = 990
    hist[51] = 10
    cutoff = threshold(hist, 1000, 0.01)
    assert sum(hist[cutoff:]) / 1000 <= 0.01
    assert sum(hist[cutoff - 1 :]) / 1000 > 0.01
    assert cutoff == 51


def test_threshold_takes_the_smallest_k_that_fits():
    """Literally the smallest.  A gap in the histogram makes that visible."""
    hist = [0] * 101
    hist[50] = 990
    hist[80] = 10
    assert threshold(hist, 1000, 0.01) == 51


def test_a_tie_at_the_top_pushes_the_threshold_past_it():
    hist = [0] * 101
    hist[50] = 998
    hist[51] = 2
    assert threshold(hist, 1000, 0.001) == 52


def test_stricter_alpha_never_lowers_the_threshold():
    hist = [0] * 101
    for k in range(40, 60):
        hist[k] = 50
    cutoffs = [threshold(hist, 1000, a) for a in (0.1, 0.01, 0.001)]
    assert cutoffs == sorted(cutoffs), cutoffs


# ==========================================================================
# The two modes, end to end
#
# These run your program the way the grader does: as a program, with a
# command line, reading the one JSON object it prints.  They are the only
# tests here that check the thing you actually hand in.
# ==========================================================================

HERE = Path(__file__).resolve().parent


def run_program(*argv: str) -> dict:
    """Run ``python synthid.py ...`` and return the JSON it printed."""
    finished = subprocess.run(
        [sys.executable, str(HERE / "synthid.py"), *argv],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    assert finished.returncode == 0, (
        f"synthid.py {' '.join(argv)} exited {finished.returncode}\n"
        f"{finished.stderr[-2000:]}"
    )
    try:
        return json.loads(finished.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            "your program must print exactly one JSON object and nothing "
            f"else.  It printed:\n{finished.stdout[:400]}"
        ) from None


def test_generate_prints_the_document_it_was_asked_for():
    out = run_program(
        "--generate", "--key", "K", "--seed", "5", "--length", "40"
    )
    assert out["mode"] == "generate"
    assert out["seed"] == 5
    assert out["watermarked"] is True
    assert out["layers"] == 30 and out["window"] == 4
    assert len(out["tokens"]) == 40, len(out["tokens"])
    assert all(token in TOKENS for token in out["tokens"])


def test_generating_twice_gives_the_same_document():
    """The whole assignment rests on this one.

    Same seed, same flags, same tokens -- on this machine, on the
    grader's, and on any machine either of us runs it on next year.  If
    this fails, nothing else you are marked on can be compared with
    anything.
    """
    argv = ("--generate", "--key", "K", "--seed", "808", "--length", "60")
    assert run_program(*argv)["tokens"] == run_program(*argv)["tokens"]


def test_a_different_seed_gives_a_different_document():
    first = run_program(
        "--generate", "--key", "K", "--seed", "1", "--length", "60"
    )
    second = run_program(
        "--generate", "--key", "K", "--seed", "2", "--length", "60"
    )
    assert first["tokens"] != second["tokens"]


def test_the_watermark_is_actually_in_the_text():
    """Score what you generated.  Watermarked text must score above 0.5.

    400 tokens at 30 layers puts the null spread at about 0.005, so a
    real watermark sits well clear of it and this test is not a close
    call. If your generated text scores at 0.5, your sampler is not
    using its g-values.
    """
    out = run_program(
        "--generate", "--key", "K", "--seed", "31", "--length", "400"
    )
    marked = mean_score(out["tokens"], "K", 4, 30)

    plain = run_program(
        "--generate",
        "--key",
        "K",
        "--seed",
        "31",
        "--length",
        "400",
        "--no-watermark",
    )
    unmarked = mean_score(plain["tokens"], "K", 4, 30)

    assert marked > 0.52, f"watermarked text scored {marked:.4f}"
    assert abs(unmarked - 0.5) < 0.03, \
        f"unwatermarked text scored {unmarked:.4f}"
    assert marked > unmarked


def test_detect_calibrates_and_judges(tmp_path_factory=None):
    """Write a corpus of our own, then detect over it.

    Unwatermarked text used for calibration, watermarked text to judge:
    at alpha = 0.1 over 40 documents most of the watermarked ones should
    be flagged, and the threshold must be an integer the numerators can
    actually reach.
    """
    corpus = HERE / "_public_test_corpus"
    corpus.mkdir(exist_ok=True)
    calibration = corpus / "cal.txt"
    documents = corpus / "docs.txt"

    def write(path: Path, seeds: range, extra: tuple[str, ...]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for seed in seeds:
                out = run_program(
                    "--generate",
                    "--key",
                    "K",
                    "--seed",
                    str(seed),
                    "--length",
                    "200",
                    *extra,
                )
                handle.write(" ".join(out["tokens"]) + "\n")

    write(calibration, range(1000, 1080), ("--no-watermark",))
    write(documents, range(2000, 2030), ())

    out = run_program(
        "--detect",
        "--key",
        "K",
        "--calibration",
        str(calibration),
        "--documents",
        str(documents),
        "--alpha",
        "0.1",
    )
    assert out["mode"] == "detect"
    assert out["calibration"]["documents"] == 80
    cutoff = out["calibration"]["threshold"]
    assert isinstance(cutoff, int)
    assert len(out["documents"]) == 30
    assert out["flagged"] == sum(e["flagged"] for e in out["documents"])
    assert out["flagged"] >= 20, (
        f"only {out['flagged']} of 30 watermarked documents were flagged"
    )
    for entry in out["documents"]:
        assert close(entry["score"],
                     entry["numerator"] / (30 * entry["scorable"]))
        assert entry["flagged"] == (entry["numerator"] >= cutoff)


def test_the_bloom_filter_costs_you_evidence():
    """The same detection, with a filter too small for the job.

    Fewer bits means more collisions, more positions wrongly skipped, and
    less evidence left to judge on.  The false skips must fall as the
    filter grows.
    """
    corpus = HERE / "_public_test_corpus"
    if not (corpus / "docs.txt").is_file():
        test_detect_calibrates_and_judges()

    skips = {}
    for bits in (512, 8192):
        out = run_program(
            "--detect",
            "--key",
            "K",
            "--calibration",
            str(corpus / "cal.txt"),
            "--documents",
            str(corpus / "docs.txt"),
            "--bloom-bits",
            str(bits),
        )
        assert out["bloom"] == {"bits": bits, "hashes": 3}
        skips[bits] = sum(e["false_skips"] for e in out["documents"])
    assert skips[512] > skips[8192], skips


# ==========================================================================
# Running without pytest
# ==========================================================================


def main() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures: list[tuple[str, str]] = []
    for name, test in tests:
        try:
            test()
        except NotImplementedError:
            failures.append((name, "not implemented yet"))
        except AssertionError as error:
            failures.append((name, f"failed: {error}"))
        except Exception as error:
            failures.append((name, f"{type(error).__name__}: {error}"))
        else:
            print(f"  pass  {name}")
    for name, reason in failures:
        print(f"  FAIL  {name}: {reason}")
    print(f"\n{len(tests) - len(failures)} of {len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
