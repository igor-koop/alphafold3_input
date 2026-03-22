"""Shared utility functions and enums.

This submodule provides utilities for chemical component generation and CCD
serialization, alignment tracing and realignment, template reindexing, and
bijective base-26 conversion of entity identifiers.

Exports:
    - :class:`Operation`: Per-residue alignment operation enum.
    - :func:`base26_encoder`: Encode a positive 1-based integer as a
        bijective base-26 identifier.
    - :func:`base26_decoder`: Decode a bijective base-26 identifier into its
        1-based integer index.
    - :func:`ccd`: Serialize one or more chemical components into a custom
        CCD definition.
    - :func:`component`: Generate and annotate an embedded molecule from
        SMILES.
    - :func:`realign`: Apply an operation trace to rewrite an A3M alignment
        into a new query coordinate system.
    - :func:`reindex`: Update template residue index mappings by lifting
        reference positions into query coordinates.
    - :func:`trace`: Compute a per-position operation trace for an A3M-style
        query aligned to a FASTA reference.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import StrEnum
from os import linesep
from re import Pattern
from typing import TYPE_CHECKING, Any

from numpy import array, dtype, int32, ndarray
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdDistGeom, rdMolDescriptors

from .template import Template

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

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
"""Validation patterns for supported sequence formats.

Keys:
    FASTA: Canonical alphabets (protein, RNA, and DNA sequences).
    A3M: Alignment alphabets (sequences with gaps and insertions).
"""

CCD_CODE: Pattern[str] = re.compile(r"[A-Z0-9][A-Z0-9-]*")
"""Validation pattern for CCD codes."""


class Operation(StrEnum):
    """Per-residue alignment operation."""

    REF = "match"
    """Exact residue match."""

    SUB = "substitution"
    """Substitution mutation."""

    INS = "insertion"
    """Insertion mutation."""

    DEL = "deletion"
    """Deletion mutation."""


def trace(reference: str, query: str) -> tuple[Operation, ...]:
    """Generate an operation trace for alignment of ``query`` to ``reference``.

    Interprets an A3M-style ``query`` sequence against a canonical FASTA
    ``reference`` and emits a per-position sequence of alignment operations.

    Args:
        reference (str): Canonical reference sequence in FASTA format.
        query (str): Aligned query sequence in A3M format.

    Returns:
        tuple[Operation, ...]: Per-position sequence of alignment operations.

    Raises:
        ValueError: If either input sequence fails validation.
    """
    if not any(pattern.fullmatch(reference) for pattern in PATTERNS["FASTA"]):
        msg = "Invalid reference sequence: must be in canonical FASTA format."
        raise ValueError(msg)

    if not any(pattern.fullmatch(query) for pattern in PATTERNS["A3M"]):
        msg = "Invalid query sequence: must be in canonical A3M format."
        raise ValueError(msg)

    out: list[Operation] = []
    reference: Iterator[str] = iter(reference)

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

    return tuple(out)


def reindex(template: Template, operations: Sequence[Operation]) -> Template:
    """Reindex template residues to a new query sequence.

    Consumes an operation trace describing how a query sequence aligns to the
    reference and updates :attr:`Template.indexes` to query coordinates.

    Args:
        template (Template): Template with indexes defined in reference
            coordinates.
        operations (Sequence[Operation]): Per-position alignment operations.

    Returns:
        Template: Template with indexes updated to query coordinates.

    Raises:
        ValueError: If an unexpected alignment operation is encountered.
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
    """Realign an A3M alignment to a new query sequence.

    Applies an operation trace describing how a query sequence aligns to the
    reference to each sequence in an A3M ``alignment``.

    Args:
        alignment (str): Sequence alignment in A3M format.
        operations (Sequence[Operation]): Per-position alignment operations.

    Returns:
        str: Realigned alignment in A3M format.

    Raises:
        ValueError: If headers or sequences are invalid, if a sequence is too
            short or too long for the operation trace, or if trailing
            insertions are encountered.
    """
    out: list[str] = []

    lines: Iterable[str] = iter(alignment.splitlines())
    for header, sequence in zip(lines, lines, strict=True):
        if not header.startswith(">"):
            msg: str = (
                f"Invalid alignment: unexpected A3M header format ({header!r})."
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
    """Construct an embedded chemical component from SMILES for CCD export.

    Parses ``smiles`` and sanitizes the molecule, adds explicit hydrogens,
    embeds a 3D conformer, optimizes its geometry, assigns deterministic atom
    names, and stores component metadata required for CCD export.

    Args:
        smiles (str): SMILES string describing a chemical component.
        code (str): Chemical component identifier.
        name (str): Human-readable component name.

    Returns:
        Mol: Embedded molecule annotated for CCD export.

    Raises:
        ValueError: If ``code`` is invalid, or if ``smiles`` is syntactically
            or chemically invalid.
        RuntimeError: If conformer embedding fails.
    """
    if not CCD_CODE.fullmatch(code):
        msg: str = (
            "Invalid chemical component code: only uppercase letters, digits "
            f"and dashes are allowed (code={code})."
        )
        raise ValueError(msg)

    if Chem.MolFromSmiles(SMILES=smiles, sanitize=False) is None:
        msg: str = (
            "Invalid chemical component SMILES: SMILES is not syntactically "
            f"valid (smiles={smiles})."
        )
        raise ValueError(msg)

    molecule: Chem.Mol | None = Chem.MolFromSmiles(
        SMILES=smiles,
        sanitize=True,
    )

    if molecule is None:
        msg: str = (
            "Invalid chemical component SMILES: SMILES does not describe a "
            f"chemically valid structure (smiles={smiles})."
        )
        raise ValueError(msg)

    molecule: Chem.Mol = Chem.AddHs(mol=molecule)

    molecule.SetProp("comp_id", code)
    molecule.SetProp("comp_name", f"{name}")
    molecule.SetProp("comp_smiles", smiles)

    atoms: defaultdict[str, int] = defaultdict(lambda: 0)

    for atom in molecule.GetAtoms():
        element: str = atom.GetSymbol().upper()
        atoms[element] += 1
        atom.SetProp("atom_name", f"{element}{atoms[element]}")

    if any(
        bond.GetBondType() == Chem.BondType.AROMATIC
        for bond in molecule.GetBonds()
    ):
        Chem.Kekulize(mol=molecule, clearAromaticFlags=True)

    parameters: rdDistGeom.EmbedParameters = AllChem.ETKDGv3()  # ty:ignore[unresolved-attribute]
    parameters.maxIterations = 500
    parameters.useRandomCoords = True

    cid: int = AllChem.EmbedMolecule(molecule, parameters)  # ty:ignore[unresolved-attribute]
    if cid < 0:
        msg: str = (
            "Invalid chemical component: conformer generation from SMILES "
            f"failed (smiles={smiles})."
        )
        raise RuntimeError(msg)

    AllChem.UFFOptimizeMolecule(molecule, maxIters=500)  # ty:ignore[unresolved-attribute]

    return molecule


def ccd(  # noqa: C901, PLR0912, PLR0915
    *components: Chem.Mol,
) -> Generator[str]:
    """Export chemical component dictionaries for one or more components.

    For each input component, yields a CCD definition containing component
    metadata, atoms, and bonds. Input molecules must provide molecule
    properties ``comp_id`` and ``comp_name`` and an ``atom_name`` property
    for every atom.

    Args:
        *components (Mol): Chemical components for CCD export.

    Yields:
        str: CCD definition of each specified component.

    Raises:
        KeyError: If required molecule or atom properties are missing.
        ValueError: If a component has no conformer or coordinates.
        TypeError: If an unsupported bond type or stereochemical
            configuration is encountered.
    """
    for component in components:
        info: dict[str, str] = {}
        atoms: defaultdict[str, list[str]] = defaultdict(list)
        bonds: defaultdict[str, list[str]] = defaultdict(list)

        for key in ("comp_id", "comp_name"):
            if not component.HasProp(key):
                msg: str = (
                    "Invalid chemical component: missing required property "
                    f"(`{key!r}`)."
                )
                raise KeyError(msg)

        try:
            conformer: Chem.Conformer = component.GetConformer(0)
        except Exception as e:
            msg: str = (
                "Invalid chemical component: no available conformers "
                f"(component={component.GetProp('comp_id')!r})."
            )
            raise ValueError(msg) from e

        coordinates: np.ndarray = conformer.GetPositions()

        if coordinates is None:
            msg: str = (
                "Invalid chemical component: conformer coordinates are "
                f"unavailable (component={component.GetProp('comp_id')!r})."
            )
            raise ValueError(msg)

        formula: str = "".join(
            " " + char if char.isupper() else char
            for char in rdMolDescriptors.CalcMolFormula(component)
        )[1:]

        info["id"] = component.GetProp("comp_id")
        info["name"] = component.GetProp("comp_name")
        info["type"] = '"NON-POLYMER"'
        info["formula"] = f'"{formula}"'
        info["formula_weight"] = f"{Descriptors.MolWt(component):.3f}"  # ty:ignore[unresolved-attribute]
        info["mon_nstd_parent_comp_id"] = "?"
        info["pdbx_synonyms"] = "?"
        info["pdbx_smiles"] = component.GetProp("comp_smiles")

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
                    msg: str = (
                        "Invalid chemical component: unsupported bond type "
                        f"(type={bond.GetBondType()}, "
                        f"component={component.GetProp('comp_id')!r})."
                    )
                    raise TypeError(msg)

            match bond.GetStereo():
                case Chem.BondStereo.STEREONONE:
                    bonds["pdbx_stereo_config"].append("N")
                case Chem.BondStereo.STEREOE | Chem.BondStereo.STEREOTRANS:
                    bonds["pdbx_stereo_config"].append("E")
                case Chem.BondStereo.STEREOZ | Chem.BondStereo.STEREOCIS:
                    bonds["pdbx_stereo_config"].append("Z")
                case _:
                    msg: str = (
                        "Invalid chemical component: unsupported bond "
                        f"stereochemistry (type={bond.GetBondType()}, "
                        f"component={component.GetProp('comp_id')!r})."
                    )
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
    bijective base-26 convention.

    Args:
        n (int): Positive 1-based integer to encode.

    Returns:
        str: Encoded identifier.

    Raises:
        ValueError: If ``n`` is less than 1.
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

    Converts an alphabetic identifier in the bijective base-26 convention into
    its corresponding 1-based integer.

    Args:
        s (str): Identifier to decode.

    Returns:
        int: Decoded positive 1-based integer.

    Raises:
        ValueError: If ``s`` is empty or contains characters outside A-Z.
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
