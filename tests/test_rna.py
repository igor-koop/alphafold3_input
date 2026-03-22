"""Tests for RNA chain entity models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest
from pydantic import ValidationError

from alphafold3_input.modification import Entity, Modification
from alphafold3_input.rna import RNA

if TYPE_CHECKING:
    from pathlib import Path


class TestRNA:
    """Tests for the ``RNA`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        rna: RNA = RNA(sequence="ACGU")

        assert rna.id is None
        assert rna.description is None
        assert rna.sequence == "ACGU"
        assert rna.modifications == ()
        assert rna.alignment is None
        assert rna.copies == 1

    def test_construct_aliases(self: Self) -> None:
        """Validate construction from wrapped RNA aliases."""
        rna: RNA = RNA.model_validate(
            {
                "rna": {
                    "id": ["A"],
                    "sequence": "ACGU",
                    "modifications": [
                        {
                            "modificationType": "2MG",
                            "basePosition": 1,
                        },
                    ],
                    "unpairedMsa": ">query\nACGU\n",
                },
            },
        )

        assert rna.id == ["A"]
        assert rna.sequence == "ACGU"
        assert len(rna.modifications) == 1
        assert rna.modifications[0].type == "2MG"
        assert rna.modifications[0].position == 1
        assert rna.alignment == ">query\nACGU\n"
        assert rna.copies == 1

    @pytest.mark.parametrize(
        "identifier",
        [
            "a",
            "Ab",
            "A1",
            "A_B",
            "A-B",
            "",
        ],
    )
    def test_validate_id(self: Self, identifier: str) -> None:
        """Validate bijective base-26 format of RNA identifiers."""
        with pytest.raises(ValidationError):
            RNA(sequence="ACGU", id=identifier)

    @pytest.mark.parametrize(
        "sequence",
        [
            "acgu",
            "ACGT",
            "ACGN",
            "A-CG",
            "",
        ],
    )
    def test_validate_sequence(self: Self, sequence: str) -> None:
        """Validate that RNA sequences contain only A, C, G, and U."""
        with pytest.raises(ValidationError):
            RNA(sequence=sequence)

    @pytest.mark.parametrize("copies", [0, -1, -5])
    def test_validate_copies(self: Self, copies: int) -> None:
        """Validate that the RNA copy count must be at least one."""
        with pytest.raises(ValidationError):
            RNA(sequence="ACGU", copies=copies)

    def test_validate_copies_derived(self: Self) -> None:
        """Validate that the copy count is derived from explicit identifiers."""
        rna: RNA = RNA(
            id=["A", "B"],
            sequence="ACGU",
        )

        assert rna.id == ["A", "B"]
        assert rna.copies == 2

    def test_validate_copies_consistency(self: Self) -> None:
        """Validate consistency between explicit copies and identifiers."""
        with pytest.raises(
            ValidationError,
            match="inconsistent with the length of `id`",
        ):
            RNA(
                id=["A", "B"],
                sequence="ACGU",
                copies=3,
            )

    def test_validate_modify(self: Self) -> None:
        """Validate that ``modify()`` appends residue modifications."""
        rna: RNA = RNA(sequence="AGCU")

        out: RNA = rna.modify(
            Modification(type="2MG", position=1),
            Modification(type="5MC", position=4),
        )

        assert out is rna
        assert len(rna.modifications) == 2
        assert rna.modifications[0].type == "2MG"
        assert rna.modifications[0].position == 1
        assert rna.modifications[1].type == "5MC"
        assert rna.modifications[1].position == 4

    def test_validate_modify_empty(self: Self) -> None:
        """Validate that ``modify()`` requires at least one modification."""
        rna: RNA = RNA(sequence="AGCU")

        with pytest.raises(TypeError, match="no modifications were provided"):
            rna.modify()

    def test_validate_modify_scope(self: Self) -> None:
        """Validate that RNA scope is assigned to all modifications."""
        rna: RNA = RNA(
            sequence="AGCU",
            modifications=(
                Modification(type="2MG", position=1),
                Modification(type="5MC", position=4),
            ),
        )

        assert all(
            modification.scope == Entity.RNA
            for modification in rna.modifications
        )

    def test_validate_modify_position(self: Self) -> None:
        """Validate that modification positions must be within the sequence."""
        with pytest.raises(ValidationError, match="out of range"):
            RNA(
                sequence="ACGU",
                modifications=(Modification(type="2MG", position=5),),
            )

    def test_validate_modify_unique(self: Self) -> None:
        """Validate that modification positions must be unique."""
        with pytest.raises(ValidationError, match="same `position`"):
            RNA(
                sequence="ACGU",
                modifications=(
                    Modification(type="2MG", position=2),
                    Modification(type="5MC", position=2),
                ),
            )

    def test_validate_alignment_inline(self: Self) -> None:
        """Validate inline alignment input."""
        rna: RNA = RNA(
            sequence="ACGU",
            alignment=">query\nACGU\n",
        )

        assert rna.alignment == ">query\nACGU\n"

    def test_validate_alignment_path(self: Self, tmp_path: Path) -> None:
        """Validate path-based alignment coercion."""
        path: Path = tmp_path / "alignment.a3m"

        rna: RNA = RNA.model_validate(
            {
                "rna": {
                    "sequence": "ACGU",
                    "unpairedMsaPath": str(path),
                },
            },
        )

        assert rna.alignment == path

    def test_serialize_alignment_inline(self: Self) -> None:
        """Validate serialization of inline alignments."""
        rna: RNA = RNA(
            sequence="ACGU",
            alignment=">query\nACGU\n",
        )

        data: dict[str, object] = rna.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "rna" in data
        assert isinstance(data["rna"], dict)
        assert data["rna"]["unpairedMsa"] == ">query\nACGU\n"
        assert "unpairedMsaPath" not in data["rna"]

    def test_serialize_alignment_path(self: Self, tmp_path: Path) -> None:
        """Validate serialization of path-based alignments."""
        path: Path = tmp_path / "alignment.a3m"
        rna: RNA = RNA(
            sequence="ACGU",
            alignment=path,
        )

        data: dict[str, object] = rna.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "rna" in data
        assert isinstance(data["rna"], dict)
        assert data["rna"]["unpairedMsaPath"] == path
        assert "unpairedMsa" not in data["rna"]

    def test_serialize(self: Self) -> None:
        """Validate wrapped serialization of RNA entities."""
        rna: RNA = RNA(
            id=["A"],
            description="Example RNA",
            sequence="ACGU",
            modifications=(Modification(type="2MG", position=1),),
        )

        data: dict[str, object] = rna.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "rna" in data
        assert isinstance(data["rna"], dict)
        assert data["rna"]["id"] == ["A"]
        assert data["rna"]["description"] == "Example RNA"
        assert data["rna"]["sequence"] == "ACGU"
        assert "modifications" in data["rna"]
        assert "copies" not in data["rna"]
