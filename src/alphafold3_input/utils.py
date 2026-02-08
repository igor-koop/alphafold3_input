"""Shared utilities.

This submodule defines bijective base-26 conversion utilities for entity
identifiers.

Exports:
    base26_encoder: Encode a positive 1-based integer as a bijective base-26
        identifier.
    base26_decoder: Decode a bijective base-26 label back into its 1-based
        integer index.
"""

from __future__ import annotations

__all__: list[str] = [
    "base26_decoder",
    "base26_encoder",
]


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

    if any(not ("A" <= c <= "Z") for c in s):
        msg: str = (
            "Invalid bijective base-26 label: expected only uppercase letters "
            f"A-Z (label={s!r})."
        )
        raise ValueError(msg)

    n = 0
    for c in s:
        n = n * 26 + (ord(c) - ord("A") + 1)

    return n
