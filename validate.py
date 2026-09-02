"""Check your program's output before you rely on it.

    python synthid.py --generate --key K --seed 7 --length 100 > out.json
    python validate.py out.json

    python synthid.py --detect ... | python validate.py -

Reads one JSON object, works out which mode produced it, and checks it
against the rules in ``output.schema.json``.

**Not valid JSON** is the fatal one: if your program printed anything
besides the single object -- a debug line, a traceback -- there is
nothing to read, and that run earns nothing.  **Errors** mean the object
is there but does not match the schema; that costs the marks for
matching the schema, and a field of the wrong shape also costs whatever
is compared through it.  **Warnings** mean it is well formed but
something in it looks wrong -- a flagged count that disagrees with the
per-document verdicts, a score that does not match its own numerator.

No third-party packages: this runs anywhere Python does.
"""

import json
import sys
from typing import Any, TypeGuard

TOKEN_LENGTH = 4
ALLOWED_ENTROPY = ("high", "low")
ALLOWED_SAMPLER = ("layered", "knockout")

GENERATE_KEYS = {
    "mode", "seed", "watermarked", "entropy", "sampler", "layers", "window", "tokens",
}  # fmt: skip
GENERATE_REQUIRED = {
    "mode", "seed", "watermarked", "layers", "window", "sampler", "tokens",
}  # fmt: skip
DETECT_KEYS = {
    "mode", "alpha", "layers", "window", "bloom", "calibration", "documents", "flagged",
}  # fmt: skip
DETECT_REQUIRED = {
    "mode", "alpha", "layers", "window", "calibration", "documents", "flagged",
}  # fmt: skip
ENTRY_KEYS = {"index", "scorable", "numerator", "score", "flagged", "false_skips"}
ENTRY_REQUIRED = {"index", "scorable", "numerator", "score", "flagged"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")

    @property
    def ok(self) -> bool:
        return not self.errors


def is_int(value: object) -> TypeGuard[int]:
    """A real integer.  ``True == 1`` in Python; the schema does not agree."""
    return isinstance(value, int) and not isinstance(value, bool)


def check_keys(
    report: Report,
    where: str,
    obj: dict[str, Any],
    allowed: set[str],
    required: set[str],
) -> None:
    for name in sorted(required - set(obj)):
        report.error(where, f'missing "{name}"')
    for name in sorted(set(obj) - allowed):
        report.error(where, f'unexpected key "{name}"')


def check_count(report: Report, where: str, value: object, minimum: int = 0) -> None:
    if not is_int(value):
        report.error(where, f"expected an integer, found {value!r}")
    elif value < minimum:
        report.error(where, f"expected at least {minimum}, found {value}")


def check_generate(report: Report, out: dict[str, Any]) -> None:
    check_keys(report, "output", out, GENERATE_KEYS, GENERATE_REQUIRED)
    check_count(report, "seed", out.get("seed"), minimum=-(2**63))
    check_count(report, "layers", out.get("layers"), minimum=1)
    check_count(report, "window", out.get("window"), minimum=1)
    if not isinstance(out.get("watermarked"), bool):
        report.error("watermarked", "expected true or false")
    if "entropy" in out and out["entropy"] not in ALLOWED_ENTROPY:
        report.error("entropy", f"expected one of {ALLOWED_ENTROPY}")
    if out.get("sampler") not in ALLOWED_SAMPLER:
        report.error("sampler", f"expected one of {ALLOWED_SAMPLER}")

    tokens = out.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        report.error("tokens", "expected a non-empty array of tokens")
        return
    for index, token in enumerate(tokens):  # pyright: ignore[reportUnknownVariableType]
        if (
            not isinstance(token, str)
            or len(token) != TOKEN_LENGTH
            or token[0] != "w"
            or not token[1:].isdigit()
        ):
            report.error(f"tokens[{index}]", f"expected w000..w999, found {token!r}")
            return


def check_detect(report: Report, out: dict[str, Any]) -> None:
    check_keys(report, "output", out, DETECT_KEYS, DETECT_REQUIRED)
    check_count(report, "layers", out.get("layers"), minimum=1)
    check_count(report, "window", out.get("window"), minimum=1)

    alpha = out.get("alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        report.error("alpha", "expected a number")
    elif not 0 < float(alpha) <= 1:
        report.error("alpha", f"expected 0 < alpha <= 1, found {alpha}")

    calibration = out.get("calibration")
    if not isinstance(calibration, dict):
        report.error("calibration", "expected an object")
    else:
        cal: dict[str, Any] = calibration
        check_keys(
            report,
            "calibration",
            cal,
            {"documents", "threshold"},
            {"documents", "threshold"},
        )
        check_count(report, "calibration.documents", cal.get("documents"), minimum=1)
        check_count(report, "calibration.threshold", cal.get("threshold"))

    if "bloom" in out:
        bloom = out["bloom"]
        if not isinstance(bloom, dict):
            report.error("bloom", "expected an object")
        else:
            b: dict[str, Any] = bloom
            check_keys(report, "bloom", b, {"bits", "hashes"}, {"bits", "hashes"})
            check_count(report, "bloom.bits", b.get("bits"), minimum=8)
            check_count(report, "bloom.hashes", b.get("hashes"), minimum=1)

    documents = out.get("documents")
    if not isinstance(documents, list):
        report.error("documents", "expected an array")
        return

    layers = out.get("layers")
    counted = 0
    entries: list[Any] = documents
    for index, entry in enumerate(entries):
        where = f"documents[{index}]"
        if not isinstance(entry, dict):
            report.error(where, "expected an object")
            continue
        item: dict[str, Any] = entry
        check_keys(report, where, item, ENTRY_KEYS, ENTRY_REQUIRED)
        check_count(report, f"{where}.index", item.get("index"))
        check_count(report, f"{where}.scorable", item.get("scorable"))
        check_count(report, f"{where}.numerator", item.get("numerator"))
        if item.get("flagged") not in (0, 1) or isinstance(item.get("flagged"), bool):
            report.error(f"{where}.flagged", "expected 0 or 1")
        elif item["flagged"] == 1:
            counted += 1
        if is_int(item.get("index")) and item["index"] != index:
            report.warn(where, f"index is {item['index']}, expected {index}")

        score = item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            report.error(f"{where}.score", "expected a number")
        elif not 0.0 <= float(score) <= 1.0:
            report.error(f"{where}.score", f"expected 0 <= score <= 1, found {score}")
        elif (
            is_int(item.get("numerator"))
            and is_int(item.get("scorable"))
            and is_int(layers)
            and layers > 0
        ):
            expected = (
                item["numerator"] / (layers * item["scorable"])
                if item["scorable"]
                else 0.0
            )
            if abs(float(score) - expected) > 1e-9:
                report.warn(
                    f"{where}.score",
                    f"{score} is not numerator/(layers*scorable) = {expected:.6f}",
                )

    flagged = out.get("flagged")
    check_count(report, "flagged", flagged)
    if is_int(flagged) and flagged != counted:
        report.warn(
            "flagged",
            f"says {flagged}, but {counted} documents carry flagged = 1",
        )


def validate(out: object) -> Report:
    report = Report()
    if not isinstance(out, dict):
        report.error("output", "the top level must be a JSON object")
        return report
    document: dict[str, Any] = out
    mode = document.get("mode")
    if mode == "generate":
        check_generate(report, document)
    elif mode == "detect":
        check_detect(report, document)
    else:
        report.error("mode", f'expected "generate" or "detect", found {mode!r}')
    return report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    source = args[0] if args else "-"
    try:
        if source == "-":
            text = sys.stdin.read()
        else:
            with open(source, encoding="utf-8") as handle:
                text = handle.read()
    except OSError as error:
        print(f"ERROR    cannot read {source}: {error}")
        return 1
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as error:
        print(f"ERROR    not valid JSON: {error}")
        print("         (your program must print the JSON object and nothing else)")
        return 1

    report = validate(loaded)
    for message in report.errors:
        print(f"ERROR    {message}")
    for message in report.warnings:
        print(f"WARNING  {message}")
    print()
    if report.errors:
        print(
            f"{len(report.errors)} error(s): this output does not match the "
            "schema, and will lose the marks for matching it."
        )
        return 1
    if report.warnings:
        print(f"Matches the schema, with {len(report.warnings)} warning(s) to look at.")
        return 0
    print("Matches the schema and looks consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
