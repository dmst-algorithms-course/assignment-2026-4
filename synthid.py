"""SynthID assignment -- your work goes here.

This file is what you submit. Your program is run in two ways::

    python synthid.py --generate --key KEY --seed 42 --length 400
    python synthid.py --detect --key KEY --calibration cal.txt --documents docs.txt

Exactly one JSON object goes to stdout, matching
``output.schema.json``.  Check it with ``python validate.py`` before you
rely on it.

Besides ``toolkit``, only these modules may be imported, and the grader
checks before it runs anything:

    array, bisect, collections, dataclasses, enum, functools, hashlib,
    heapq, itertools, json, math, operator, random, string, typing

Anything else means a zero for the parts of the mark that come from
running your code.

"""

from collections.abc import Sequence
from typing import Any

# These are the pieces of toolkit.py you are most likely to want.
from toolkit import (  # noqa: F401
    DEFAULT_ALPHA,
    DEFAULT_LAYERS,
    DEFAULT_WINDOW,
    MODEL_SEED,
    PROFILES,
    TOKENS,
    VOCAB_SIZE,
    Model,
    Randomness,
    bloom_hash_positions,
    choose,
    context_item,
    g_value,
    read_documents,
    run,
)

# --------------------------------------------------------------------------
# Task A -- two implementations of one sampler
# --------------------------------------------------------------------------


def sample_layered(
    p: Sequence[float], g: Sequence[Sequence[int]], m: int
) -> list[float]:
    """Return the distribution the tournament draws from, after m layers.

    This returns a distribution*: a list of probabilities the same length
    as ``p``, summing to 1.  It does not return a token.
    """
    raise NotImplementedError


def sample_knockout(
    p: Sequence[float], g: Sequence[Sequence[int]], m: int, rng: Randomness
) -> int:
    """Draw one token by running the tournament for real."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Task B -- the detector
# --------------------------------------------------------------------------


def scored_positions(tokens: Sequence[str], h: int) -> list[int]:
    """Which positions of a document the detector scores, in order."""
    raise NotImplementedError


def score_numerator(tokens: Sequence[str], key: str, h: int, m: int) -> int:
    """The integer k: g-values summed over scored positions and layers."""
    raise NotImplementedError


def mean_score(tokens: Sequence[str], key: str, h: int, m: int) -> float:
    """ The mean score for the document."""
    raise NotImplementedError


class BloomFilter:
    """A fixed-size stand-in for the seen-set.

    Store the bits *packed*: a ``bytearray(nbits // 8)``, or one ``int``
    used as a bitmask.
    """

    def __init__(self, nbits: int, k: int) -> None:
        raise NotImplementedError

    def add(self, item: str) -> None:
        """Set every bit of ``item``, using ``bloom_hash_positions``."""
        raise NotImplementedError

    def contains(self, item: str) -> bool:
        """Are all of ``item``'s bits set?"""
        raise NotImplementedError


def bloom_scored_positions(
    tokens: Sequence[str], h: int, nbits: int, k: int
) -> list[int]:
    """``scored_positions`` with a Bloom filter instead of the set."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Task C -- calibration
# --------------------------------------------------------------------------


def histogram(numerators: Sequence[int], length: int, m: int) -> list[int]:
    """Tally the numerators: ``hist[k]`` documents scored exactly k."""
    raise NotImplementedError


def threshold(hist: Sequence[int], n_docs: int, alpha: float) -> int:
    """The smallest k with ``sum(hist[j] for j >= k) / n_docs <= alpha``"""
    raise NotImplementedError


# --------------------------------------------------------------------------
# The two modes
# --------------------------------------------------------------------------


def generate(
    *,
    key: str,
    seed: int,
    length: int,
    layers: int,
    window: int,
    entropy: str,
    watermarked: bool,
    sampler: str,
) -> dict[str, Any]:
    """Write one document and return the JSON object to print."""
    raise NotImplementedError


def detect(
    *,
    key: str,
    calibration: str,
    documents: str,
    alpha: float,
    layers: int,
    window: int,
    bloom_bits: int | None,
    bloom_hashes: int,
) -> dict[str, Any]:
    """Calibrate on unwatermarked text, judge documents, return the JSON."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(run(generate, detect))
