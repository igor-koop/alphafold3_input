"""Shared utilities.

This submodule defines TODO, alignment utilities for computing sequence variant
liftover, redindexing of templates, and realignment of sequence alignments, as
well as provides bijective base-26 conversion utilities for entity identifiers.

Exports:
    Operation: TODO.
    base26_encoder: Encode a positive 1-based integer as a bijective base-26
        identifier.
    base26_decoder: Decode a bijective base-26 label back into its 1-based
        integer index.
    ccd: TODO.
    component: TODO.
    realign: Apply an operation trace to rewrite an A3M alignment into the new
        query coordinate system.
    reindex: Update template residue index mappings by lifting reference
        positions into query coordinates.
    trace: Compute a per-position operation trace for an A3M-style query
        aligned to a FASTA reference.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import StrEnum
from os import linesep
from typing import TYPE_CHECKING, Any

from numpy import array, dtype, int32, ndarray
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdDistGeom, rdMolDescriptors

from .template import Template

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    import numpy as np

__all__: list[str] = [
    "Operation",
    "base26_decoder",
    "base26_encoder",
    "ccd",
    "component",
    "realign",
    "reindex",
    "trace",
]

PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
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

    lines: Iterable[str] = iter(alignment.splitlines())
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

    return linesep.join(out)


def component(
    smiles: str,
    code: str,
    name: str,
) -> Chem.Mol:
    """TODO."""
    molecule: Chem.Mol | None = Chem.MolFromSmiles(
        SMILES=smiles,
        sanitize=False,
    )

    if molecule is None:
        msg: str = "SMILES is not syntactically valid."
        raise ValueError(msg)

    try:
        Chem.SanitizeMol(mol=molecule)
    except Exception as e:
        msg: str = "SMILES does not describe a chemically valid structure."
        raise ValueError(msg) from e

    try:
        Chem.Kekulize(mol=molecule)
    except Exception as e:
        msg: str = "SMILES could not be converted into kekulized form."
        raise ValueError(msg) from e

    molecule: Chem.Mol = Chem.AddHs(mol=molecule)

    parameters: rdDistGeom.EmbedParameters = AllChem.ETKDGv3()  # ty:ignore[unresolved-attribute]
    parameters.maxIterations = 500
    parameters.useRandomCoords = True

    cid: int = AllChem.EmbedMolecule(molecule, parameters)  # ty:ignore[unresolved-attribute]
    if cid < 0:
        msg: str = "Conformer generation from SMILES failed."
        raise RuntimeError(msg)

    AllChem.UFFOptimizeMolecule(molecule, maxIters=500)  # ty:ignore[unresolved-attribute]

    atoms: defaultdict[str, int] = defaultdict(lambda: 0)

    for atom in molecule.GetAtoms():
        element: str = atom.GetSymbol().upper()
        atoms[element] += 1
        atom.SetProp("atom_name", f"{element}{atoms[element]}")

    molecule.SetProp("comp_id", code)
    molecule.SetProp("comp_name", name)

    return molecule


def ccd(  # noqa: C901, PLR0912
    *components: Chem.Mol,
) -> Generator[str]:
    """TODO."""
    for component in components:
        info: dict[str, str] = {}
        atoms: defaultdict[str, list[str]] = defaultdict(list)
        bonds: defaultdict[str, list[str]] = defaultdict(list)

        coordinates: np.ndarray = component.GetConformer(0).GetPositions()

        if coordinates is None:
            msg: str = ""
            raise ValueError

        info["id"] = component.GetProp("comp_id")
        info["name"] = component.GetProp("comp_name")
        info["type"] = "NON-POLYMER"
        info["formula"] = rdMolDescriptors.CalcMolFormula(component)
        info["formula_weight"] = f"{Descriptors.MolWt(component):.3f}"  # ty:ignore[unresolved-attribute]
        info["mon_nstd_parent_comp_id"] = "?"
        info["pdbx_synonyms"] = "?"

        for atom in component.GetAtoms():
            atoms["comp_id"].append(info["id"])
            atoms["atom_id"].append(atom.GetProp("atom_name"))
            atoms["type_symbol"].append(atom.GetSymbol().upper())
            atoms["charge"].append(str(int(atom.GetFormalCharge())))

        for idx, dimension in enumerate(("x", "y", "z")):
            atoms[f"pdbx_model_Cartn_{dimension}_ideal"] = [
                f"{coordinate:.3f}" for coordinate in coordinates[:, idx]
            ]

        for bond in component.GetBonds():
            bonds["comp_id"].append(info["id"])
            bonds["atom_id_1"].append(bond.GetBeginAtom().GetProp("atom_name"))
            bonds["atom_id_2"].append(bond.GetEndAtom().GetProp("atom_name"))

            match bond.GetBondType():
                case Chem.BondType.DATIVE | Chem.BondType.SINGLE:
                    bonds["value_order"].append("SING")
                case Chem.BondType.DOUBLE:
                    bonds["value_order"].append("DOUB")
                case Chem.BondType.TRIPLE:
                    bonds["value_order"].append("TRIP")
                case _:
                    msg: str = ""
                    raise TypeError(msg)

            match bond.GetStereo():
                case Chem.BondStereo.STEREONONE:
                    bonds["pdbx_stereo_config"].append("N")
                case Chem.BondStereo.STEREOE | Chem.BondStereo.STEREOTRANS:
                    bonds["pdbx_stereo_config"].append("E")
                case Chem.BondStereo.STEREOZ | Chem.BondStereo.STEREOCIS:
                    bonds["pdbx_stereo_config"].append("Z")
                case _:
                    msg: str = ""
                    raise TypeError(msg)

            bonds["pdbx_aromatic_flag"].append(
                "Y" if bond.GetIsAromatic() else "N",
            )

        definition: list[list[str]] = [
            [f"data_{info['id']}"],
            ["#"],
            [f"_chem_comp.{key} {val}" for key, val in info.items()],
            ["#"],
            ["loop_"],
            [f"_chem_comp_atom.{key}" for key in atoms],
            [" ".join(row) for row in zip(*atoms.values(), strict=True)],
            ["#"],
            ["loop_"],
            [f"_chem_comp_bond.{key}" for key in bonds],
            [" ".join(row) for row in zip(*bonds.values(), strict=True)],
            ["#"],
        ]

        yield linesep.join(line for block in definition for line in block)


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
