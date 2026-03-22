"""Tests for protein chain entity models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest
from pydantic import ValidationError

from alphafold3_input.modification import Entity, Modification
from alphafold3_input.protein import Protein
from alphafold3_input.template import Template

if TYPE_CHECKING:
    from pathlib import Path


class TestProtein:
    """Tests for the ``Protein`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        protein: Protein = Protein(sequence="ACDE")

        assert protein.id is None
        assert protein.description is None
        assert protein.sequence == "ACDE"
        assert protein.modifications == ()
        assert protein.alignment is None
        assert protein.templates is None
        assert protein.copies == 1

    def test_construct_aliases(self: Self) -> None:
        """Validate construction from wrapped protein aliases."""
        protein: Protein = Protein.model_validate(
            {
                "protein": {
                    "description": "Example protein",
                    "sequence": "ACDE",
                    "modifications": [
                        {
                            "ptmType": "HY3",
                            "ptmPosition": 1,
                        },
                    ],
                    "unpairedMsa": ">query\nACDE\n",
                },
            },
        )

        assert protein.id is None
        assert protein.description == "Example protein"
        assert protein.sequence == "ACDE"
        assert len(protein.modifications) == 1
        assert protein.modifications[0].type == "HY3"
        assert protein.modifications[0].position == 1
        assert protein.alignment == ">query\nACDE\n"
        assert protein.templates is None
        assert protein.copies == 1

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
        """Validate bijective base-26 format of protein identifiers."""
        with pytest.raises(ValidationError):
            Protein(sequence="ACDE", id=identifier)

    @pytest.mark.parametrize(
        "sequence",
        [
            "acde",
            "ACDU",
            "ACDE*",
            "",
        ],
    )
    def test_validate_sequence(self: Self, sequence: str) -> None:
        """Validate that protein sequences use standard amino acid codes."""
        with pytest.raises(ValidationError):
            Protein(sequence=sequence)

    @pytest.mark.parametrize("copies", [0, -1, -5])
    def test_validate_copies(self: Self, copies: int) -> None:
        """Validate that the protein copy count must be at least one."""
        with pytest.raises(ValidationError):
            Protein(sequence="ACDE", copies=copies)

    def test_validate_copies_derived(self: Self) -> None:
        """Validate that the copy count is derived from explicit identifiers."""
        protein: Protein = Protein(
            id=["A", "B"],
            sequence="ACDE",
        )

        assert protein.id == ["A", "B"]
        assert protein.copies == 2

    def test_validate_copies_consistency(self: Self) -> None:
        """Validate consistency between explicit copies and identifiers."""
        with pytest.raises(
            ValidationError,
            match="inconsistent with the length of `id`",
        ):
            Protein(
                id=["A", "B"],
                sequence="ACDE",
                copies=3,
            )

    def test_validate_modify(self: Self) -> None:
        """Validate that ``modify()`` appends residue modifications."""
        protein: Protein = Protein(sequence="PVLSCGEWQL")

        out: Protein = protein.modify(
            Modification(type="HY3", position=1),
            Modification(type="P1L", position=5),
        )

        assert out is protein
        assert len(protein.modifications) == 2
        assert protein.modifications[0].type == "HY3"
        assert protein.modifications[0].position == 1
        assert protein.modifications[1].type == "P1L"
        assert protein.modifications[1].position == 5

    def test_validate_modify_empty(self: Self) -> None:
        """Validate that ``modify()`` requires at least one modification."""
        protein: Protein = Protein(sequence="PVLSCGEWQL")

        with pytest.raises(TypeError, match="no modifications were provided"):
            protein.modify()

    def test_validate_modify_scope(self: Self) -> None:
        """Validate that protein scope is assigned to all modifications."""
        protein: Protein = Protein(
            sequence="PVLSCGEWQL",
            modifications=(
                Modification(type="HY3", position=1),
                Modification(type="P1L", position=5),
            ),
        )

        assert all(
            modification.scope == Entity.PROTEIN
            for modification in protein.modifications
        )

    def test_validate_modify_position(self: Self) -> None:
        """Validate that modification positions must be within the sequence."""
        with pytest.raises(ValidationError, match="out of range"):
            Protein(
                sequence="ACDE",
                modifications=[Modification(type="HY3", position=5)],
            )

    def test_validate_modify_unique(self: Self) -> None:
        """Validate that modification positions must be unique."""
        with pytest.raises(ValidationError, match="same `position`"):
            Protein(
                sequence="ACDE",
                modifications=(
                    Modification(type="HY3", position=2),
                    Modification(type="P1L", position=2),
                ),
            )

    def test_validate_alignment_inline(self: Self) -> None:
        """Validate inline alignment input."""
        protein: Protein = Protein(
            sequence="ACDE",
            alignment=">query\nACDE\n",
        )

        assert protein.alignment == ">query\nACDE\n"

    def test_validate_alignment_path(self: Self, tmp_path: Path) -> None:
        """Validate path-based alignment coercion."""
        path: Path = tmp_path / "alignment.a3m"

        protein: Protein = Protein.model_validate(
            {
                "protein": {
                    "sequence": "ACDE",
                    "unpairedMsaPath": str(path),
                },
            },
        )

        assert protein.alignment == path

    def test_validate_templates(self: Self, tmp_path: Path) -> None:
        """Validate structural template input."""
        structure: Path = tmp_path / "template.cif"

        protein: Protein = Protein(
            sequence="RPACQLW",
            templates=(
                Template(
                    structure=structure,
                    indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
                ),
            ),
        )

        assert protein.templates is not None
        assert len(protein.templates) == 1
        assert protein.templates[0].structure == structure

    def test_serialize_alignment_inline(self: Self) -> None:
        """Validate serialization of inline alignments."""
        protein: Protein = Protein(
            sequence="ACDE",
            alignment=">query\nACDE\n",
        )

        data: dict[str, object] = protein.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "protein" in data
        assert isinstance(data["protein"], dict)
        assert data["protein"]["unpairedMsa"] == ">query\nACDE\n"
        assert data["protein"]["pairedMsa"] == ""
        assert "unpairedMsaPath" not in data["protein"]

    def test_serialize_alignment_path(self: Self, tmp_path: Path) -> None:
        """Validate serialization of path-based alignments."""
        path: Path = tmp_path / "alignment.a3m"
        protein: Protein = Protein(
            sequence="ACDE",
            alignment=path,
        )

        data: dict[str, object] = protein.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "protein" in data
        assert isinstance(data["protein"], dict)
        assert data["protein"]["unpairedMsaPath"] == path
        assert data["protein"]["pairedMsa"] == ""
        assert "unpairedMsa" not in data["protein"]

    def test_serialize_templates(self: Self, tmp_path: Path) -> None:
        """Validate serialization of structural templates."""
        structure: Path = tmp_path / "template.cif"
        protein: Protein = Protein(
            sequence="RPACQLW",
            templates=(
                Template(
                    structure=structure,
                    indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
                ),
            ),
        )

        data: dict[str, object] = protein.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "protein" in data
        assert isinstance(data["protein"], dict)
        assert "templates" in data["protein"]
        assert len(data["protein"]["templates"]) == 1

    def test_serialize(self: Self) -> None:
        """Validate wrapped serialization of protein entities."""
        protein: Protein = Protein(
            id=["A"],
            description="Example protein",
            sequence="ACDE",
            modifications=(Modification(type="HY3", position=1),),
        )

        data: dict[str, object] = protein.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "protein" in data
        assert isinstance(data["protein"], dict)
        assert data["protein"]["id"] == ["A"]
        assert data["protein"]["description"] == "Example protein"
        assert data["protein"]["sequence"] == "ACDE"
        assert "modifications" in data["protein"]
        assert "copies" not in data["protein"]
