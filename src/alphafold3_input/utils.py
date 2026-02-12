"""Shared utilities.

This submodule defines alignment utilities for computing sequence variant
liftover, redindexing of templates, and realignment of sequence alignments, as
well as provides bijective base-26 conversion utilities for entity identifiers.

Exports:
    base26_encoder: Encode a positive 1-based integer as a bijective base-26
        identifier.
    base26_decoder: Decode a bijective base-26 label back into its 1-based
        integer index.
    realign: Apply an operation trace to rewrite an A3M alignment into the new
        query coordinate system.
    reindex: Update template residue index mappings by lifting reference
        positions into query coordinates.
    trace: Compute a per-position operation trace for an A3M-style query
        aligned to a FASTA reference.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from numpy import array, dtype, int32, ndarray

from .template import Template

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__: list[str] = [
    "base26_decoder",
    "base26_encoder",
    "realign",
    "reindex",
    "trace",
]

PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    #     """Validation patterns for supported sequence formats.
    #
    #     Keys:
    #         FASTA: Canonical alphabets (protein, RNA, and DNA sequences).
    #         A3M: Alignment alphabets (sequences with gaps and insertions).
    #    """
    "FASTA": (
        re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+"),
        re.compile(r"[ACGU]+"),
        re.compile(r"[ACGT]+"),
    ),
    "A3M": (
        re.compile(r"[ACDEFGHIKLMNPQRSTVWYX-]+", re.IGNORECASE),
        re.compile(r"[ACGUX-]+", re.IGNORECASE),
        re.compile(r"[ACGTX-]+", re.IGNORECASE),
    ),
}


class Operation(StrEnum):
    """Per-residue alignment operation.

    Members:
        REF: Exact residue match (`"match"`).
        SUB: Substitution mutation (`"substitution"`).
        INS: Insertion mutation (`"insertion"`).
        DEL: Deletion mutation(`"deletion"`).
    """

    REF = "match"
    SUB = "substitution"
    INS = "insertion"
    DEL = "deletion"


def trace(reference: str, query: str) -> tuple[Operation, ...]:
    """Generate operation trace for alignment of `query` to `reference`.

    This interprets an A3M-style `query` string against a canonical FASTA
    `reference` and emits a per-residue sequence of alignment operations.

    Args:
        reference (str): Canonical reference sequence in FASTA format.
        query (str): Aligned query sequence in A3M format.

    Returns:
        out (tuple[Operation, ...]): Per-residue sequence of alignment
            operations.

    Raises:
        ValueError: If either input sequence fails validation against expected
            format.

    """
    if not any(pattern.fullmatch(reference) for pattern in PATTERNS["FASTA"]):
        msg = "Invalid reference sequence: must be in canonical FASTA format."
        raise ValueError(msg)

    if not any(pattern.fullmatch(query) for pattern in PATTERNS["A3M"]):
        msg = "Invalid query sequence: must be in canonical A3M format."
        raise ValueError(msg)

    out: list[Operation] = []
    reference = iter(reference)

    for residue in query:
        if not residue.isalpha():
            out.append(Operation.DEL)
            next(reference)
            continue

        if residue.islower():
            out.append(Operation.INS)
            continue

        if residue.isupper():
            if residue == next(reference):
                out.append(Operation.REF)
            else:
                out.append(Operation.SUB)
            continue

        msg: str = (
            f"Invalid query sequence: unexpected query residue ({residue!r})."
        )
        raise ValueError(msg)

    return tuple(out)


def reindex(template: Template, operations: Sequence[Operation]) -> Template:
    """Reindex template residues to a new query sequence.

    This consumes a trace of `operations` that describes how a query sequence
    aligns to the reference. It computes a liftover mapping from reference
    positions to query positions, then updates `template.indexes` by
    transforming each referenced position into the query coordinate system.

    Args:
        template (Template): Template object with `indexes` defined in
            reference coordinates (0-based).
        operations (Sequence[Operation]): Per-residue alignment operations,
            such as those emitted by `trace()`.

    Returns:
        out (Template): Template with `indexes` updated to query coordinates.

    Raises:
        ValueError: If an unexpected operation is encountered.

    """
    liftover: dict[int, int] = {}
    positions: ndarray[tuple[Any, ...], dtype[int32]] = array(
        [0, 0],
        dtype=int32,
    )

    for operation in operations:
        match operation:
            case Operation.REF | Operation.SUB:
                liftover[positions[0]] = int(positions[1])
                positions += (1, 1)
            case Operation.DEL:
                positions += (1, 0)
            case Operation.INS:
                positions += (0, 1)
            case _:
                msg = f"Unexpected alignment operation: {operation!r}."
                raise ValueError(msg)

    indexes: dict[int, int] = {
        liftover[reference]: structure
        for reference, structure in template.indexes.items()
        if reference in liftover
    }
    return template.model_copy(update={"indexes": indexes})


def realign(  # noqa: C901
    alignment: str,
    operations: Sequence[Operation],
) -> str:
    """Realigns an A3M alignment to a new query sequence.

    This applies a trace of `operations`, describing how a query sequence
    aligns to the reference, to each sequence of an A3M `alignment`.

    Args:
        alignment (str): Sequence alignment in A3M format.
        operations (Sequence[Operation]): Per-residue alignment operations,
            such as those emitted by `trace()`.

    Returns:
        out (str): Realigned sequence alignment in A3M format.

    Raises:
        ValueError: If headers or sequences of the alignment fail validation
            against expected; if a sequence is of unexpected length for
            provided operations trace; if a sequence contains trailing
            insertions.

    """
    out: list[str] = []

    lines = iter(alignment.splitlines())
    for header, sequence in zip(lines, lines, strict=True):
        if not header.startswith(">"):
            msg: str = (
                "Invalid alignment: unexpected A3M header format "
                f"({header!r})."
            )
            raise ValueError(msg)

        if not any(pattern.fullmatch(sequence) for pattern in PATTERNS["A3M"]):
            msg: str = (
                "Invalid alignment: unexpected A3M sequence format "
                f"({sequence!r})."
            )
            raise ValueError(msg)

        it: Iterable[str] = iter(sequence)
        seq: list[str] = []

        for operation in operations:
            if operation is Operation.INS:
                seq.append("-")
                continue

            residue: str | None = next(it, None)
            if residue is None:
                msg: str = (
                    "Invalid alignment: sequence ended too early for "
                    f"operations trace ({header!r})."
                )
                raise ValueError(msg)

            while residue.islower():
                seq.append(residue)
                residue = next(it, None)
                if residue is None:
                    msg: str = (
                        "Invalid alignment: sequence contains trailing "
                        f"insertion(s) ({header!r})."
                    )
                    raise ValueError(msg)

            if operation is Operation.DEL:
                if residue.isalpha():
                    seq.append(residue.lower())
                continue

            seq.append(residue)

        leftover: str | None = next(it, None)
        if leftover is not None:
            msg: str = (
                "Invalid alignment: sequence longer than operations trace "
                f"({header!r})."
            )
            raise ValueError(msg)

        out.append(header)
        out.append("".join(seq))

    return "\n".join(out)


def base26_encoder(n: int) -> str:
    """Encode a positive integer using bijective base-26.

    Converts a 1-based integer into an alphabetic identifier using the
    bijective base-26 convention (A-Z, then AA, AB, ...).

    Args:
        n (int): Positive 1-based integer to encode.

    Returns:
        out (str): Encoded identifier (e.g., 1 -> "A", 26 -> "Z", 27 -> "AA").

    Raises:
        ValueError: If `n` is not a positive 1-based integer (`n < 1`).

    """
    if n < 1:
        msg: str = (
            "Invalid bijective base-26 index: expected a positive 1-based "
            f"integer (n={n})."
        )
        raise ValueError(msg)

    out: list[str] = []
    while n:
        n, r = divmod(n - 1, 26)
        out.append(chr(ord("A") + r))

    return "".join(reversed(out))


def base26_decoder(s: str) -> int:
    """Decode a bijective base-26 identifier into an integer.

    Converts an alphabetic identifier in the bijective base-26 convention
    (A-Z, AA, AB, ...) into its corresponding 1-based integer.

    Args:
        s (str): Identifier to decode. Must consist only of uppercase letters
            A-Z.

    Returns:
        n (int): Decoded positive 1-based integer (e.g., "A" -> 1, "Z" -> 26,
            "AA" -> 27).

    Raises:
        ValueError: If `s` is empty or contains characters outside A-Z.

    """
    if not s:
        msg: str = (
            "Invalid bijective base-26 label: expected a non-empty string of "
            "uppercase letters A-Z."
        )
        raise ValueError(msg)

    if any(not (c.isalpha() and c.isupper()) for c in s):
        msg: str = (
            "Invalid bijective base-26 label: expected only uppercase letters "
            f"A-Z (label={s!r})."
        )
        raise ValueError(msg)

    n = 0
    for c in s:
        n = n * 26 + (ord(c) - ord("A") + 1)

    return n
