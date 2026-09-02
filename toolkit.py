"""Everything the SynthID assignment gives you.  Do not change this file.

You will not submit ``toolkit.py``, we already have it. It holds the
parts of the program that are not the assignment: the vocabulary, a
deterministic source of randomness, a toy language model standing in
for an LLM, the three hash formulas from the handout, and the command
line.

Import what you need from here into ``synthid.py``::

    from toolkit import TOKENS, Randomness, Model, g_value, run
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

# ==========================================================================
# The vocabulary
# ==========================================================================

VOCAB_SIZE = 1000
TOKENS: tuple[str, ...] = tuple(f"w{i:03d}" for i in range(VOCAB_SIZE))

#: The toy model everyone shares, so a document depends only on the seed
#: and the settings,  not on who ran it.
MODEL_SEED = "5ynth1dc0urse2026"

#: Defaults, matching the SynthID-Text paper.
DEFAULT_LAYERS = 30
DEFAULT_WINDOW = 4
DEFAULT_ALPHA = 0.01


# ==========================================================================
# Deterministic randomness
# ==========================================================================


class Randomness:
    """An endless stream of bytes, fixed by the labels it is created with.

    Two streams built from the same labels will always produce the
    same bytes.

    The three methods are named after the ones in Python's ``random``
    module, and behave the same way, so you can read a line that uses
    one without looking anything up. The difference is where the
    randomness comes from: ``random`` is seeded once, invisibly, and
    keeps its state in a module; a ``Randomness`` is seeded by the
    labels you hand it and keeps its state in the object. That is what
    makes a document reproducible. ``Randomness("generate", 42)``
    gives the same bytes everywhere, always.
    """

    def __init__(self, *parts: object) -> None:
        self._label = "|".join(str(part) for part in parts).encode()
        self._counter = 0
        self._buffer = b""
        self._position = 0

    def randbytes(self, count: int) -> bytes:
        """The next ``count`` bytes, as ``random.randbytes`` gives."""
        out = bytearray()
        while len(out) < count:
            if self._position >= len(self._buffer):
                seed = self._label + b"|" + str(self._counter).encode()
                self._buffer = sha256(seed).digest()
                self._counter += 1
                self._position = 0
            chunk = self._buffer[
                self._position : self._position + count - len(out)
            ]
            out += chunk
            self._position += len(chunk)
        return bytes(out)

    def randrange(self, bound: int) -> int:
        """An integer in ``range(bound)``, as ``random.randrange`` gives."""
        return int.from_bytes(self.randbytes(8), "big") % bound

    def random(self) -> float:
        """A float in ``[0, 1)``, as ``random.random`` gives."""

        # Divide by number of distinct values randbytes(8) can produce, 2^{64}
        return int.from_bytes(self.randbytes(8), "big") / 18446744073709551616.0


# ==========================================================================
# The toy language model
#
# A real system has an LLM here.  This stands in for one: it turns a
# context window into a distribution over the vocabulary.  The same
# context always gives the same distribution.
# ==========================================================================


@dataclass(frozen=True)
class EntropyProfile:
    """How much freedom the model has when it picks the next token.

    ``support`` tokens get non-zero probability, and the r-th of them
    gets weight proportional to ``1 / r**alpha``. A larger ``alpha``
    means a more peaked distribution -- less freedom, and so less room
    for a watermark to live in.
    """

    name: str
    support: int
    alpha: int


HIGH_ENTROPY = EntropyProfile(name="high", support=48, alpha=3)
LOW_ENTROPY = EntropyProfile(name="low", support=16, alpha=8)
PROFILES = {"high": HIGH_ENTROPY, "low": LOW_ENTROPY}


@dataclass(frozen=True)
class Distribution:
    """A next-token distribution, stored sparsely.

    ``indices[j]`` is a token index and ``probs[j]`` its probability;
    every token outside ``indices`` has probability zero.
    """

    indices: tuple[int, ...]
    probs: tuple[float, ...]


class Model:
    """The toy LLM: a context window in, a distribution out."""

    def __init__(self, model_seed: str, profile: EntropyProfile) -> None:
        self.profile = profile
        self._seed = model_seed
        weights = [
            1.0 / float(r**profile.alpha)
            for r in range(1, profile.support + 1)
        ]
        total = sum(weights)
        self._probs: tuple[float, ...] = tuple(w / total for w in weights)

    def distribution(self, window: Sequence[str]) -> Distribution:
        """What the model would say comes next, after ``window``."""
        
        rng = Randomness(self._seed, "|".join(window))
        seen: set[int] = set()
        indices: list[int] = []
        while len(indices) < len(self._probs):
            candidate = rng.randrange(VOCAB_SIZE)
            if candidate not in seen:
                seen.add(candidate)
                indices.append(candidate)
        return Distribution(indices=tuple(indices), probs=self._probs)


# ==========================================================================
# The three formulas from the handout
# ==========================================================================


def g_value(key: str, window: Sequence[str], layer: int, token: str) -> int:
    """The watermark's coin flip for one token, in one layer.  0 or 1."""
    
    payload = f"{key}|{'|'.join(window)}|{layer}|{token}".encode()
    return int.from_bytes(sha256(payload).digest(), "big") & 1


def context_item(window: Sequence[str]) -> str:
    """The context window as one string: the key of the seen-set.

    The window alone, not the token that followed: a position is a
    repeat when its context has been seen before, whatever came next.
    """
    
    return "|".join(window)


def bloom_hash_positions(item: str, nbits: int, k: int) -> list[int]:
    """The k bit positions of ``item`` in a filter of ``nbits`` bits.

    The list is always k long, but the positions in it may repeat: two
    of the hashes can land on the same bit, and then the item sets
    fewer than k bits. A Bloom filter is allowed false positives but a
    false negative means it is broken.
    """
    
    payload = item.encode()
    return [
        int.from_bytes(sha256(bytes([i]) + payload).digest(), "big") % nbits
        for i in range(k)
    ]


def choose(probs: Sequence[float], u: float) -> int:
    """Which index does ``u`` land on?  Inverse-CDF sampling.

    Walk the cumulative sum of ``probs`` and return the first index
    where it passes ``u``, a float from ``rng.random()``. Every token
    this program draws is drawn with this function, so that two runs
    agree right down to the last token: with the same ``probs`` and
    the same ``u`` it always returns the same index.
    """
    
    total = 0.0
    for index, probability in enumerate(probs):
        total += probability
        if u < total:
            return index
    return len(probs) - 1  # only reachable through float round-off


def read_documents(path: str) -> list[list[str]]:
    """One document per line, tokens separated by spaces."""
    
    with open(path, encoding="utf-8") as handle:
        return [line.split() for line in handle if line.strip()]


# ==========================================================================
# The command line
#
# Your ``synthid.py`` calls ``run(generate, detect)``; everything about
# parsing and printing is handled here, so the flags cannot drift.
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthid.py",
        description="Watermark a document, or detect a watermark"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--detect", action="store_true")

    parser.add_argument("--key", required=True)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--entropy", choices=["high", "low"], default="high")
    parser.add_argument(
        "--sampler", choices=["layered", "knockout"], default="layered"
    )
    parser.add_argument("--no-watermark", action="store_true")

    parser.add_argument("--calibration")
    parser.add_argument("--documents")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--bloom-bits", type=int, default=None)
    parser.add_argument("--bloom-hashes", type=int, default=3)
    return parser


def run(
    generate: Callable[..., dict[str, Any]],
    detect: Callable[..., dict[str, Any]],
    argv: Sequence[str] | None = None,
) -> int:
    """Parse the command line, call your function, print its JSON."""
    args = build_parser().parse_args(argv)
    if args.generate:
        if args.seed is None or args.length is None:
            print("--generate needs --seed and --length", file=sys.stderr)
            return 2
        output = generate(
            key=args.key,
            seed=args.seed,
            length=args.length,
            layers=args.layers,
            window=args.window,
            entropy=args.entropy,
            watermarked=not args.no_watermark,
            sampler=args.sampler,
        )
    else:
        if args.calibration is None or args.documents is None:
            print(
                "--detect needs --calibration and --documents",
                file=sys.stderr
            )
            return 2
        output = detect(
            key=args.key,
            calibration=args.calibration,
            documents=args.documents,
            alpha=args.alpha,
            layers=args.layers,
            window=args.window,
            bloom_bits=args.bloom_bits,
            bloom_hashes=args.bloom_hashes,
        )
    print(json.dumps(output, indent=2))
    return 0
