"""Tests for DNA chain entity models."""

from __future__ import annotations

from typing import Self

import pytest
from pydantic import ValidationError

from alphafold3_input.dna import DNA
from alphafold3_input.modification import Entity, Modification


class TestDNA:
    """Tests for the ``DNA`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        dna: DNA = DNA(sequence="GAATTC")

        assert dna.id is None
        assert dna.description is None
        assert dna.sequence == "GAATTC"
        assert dna.modifications == ()
        assert dna.copies == 1

    def test_construct_wrapped(self: Self) -> None:
        """Validate construction from a wrapped sequence mapping."""
        dna: DNA = DNA.model_validate(
            {
                "dna": {
                    "sequence": "GAATTC",
                    "description": "EcoRI site",
                },
            },
        )

        assert dna.id is None
        assert dna.description == "EcoRI site"
        assert dna.sequence == "GAATTC"
        assert dna.modifications == ()
        assert dna.copies == 1

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
        """Validate bijective base-26 format of DNA chain identifiers."""
        with pytest.raises(ValidationError):
            DNA(sequence="GAATTC", id=identifier)

    @pytest.mark.parametrize(
        "sequence",
        [
            "gaattc",
            "GAU",
            "GAN",
            "GA-TTC",
            "",
        ],
    )
    def test_validate_sequence(self: Self, sequence: str) -> None:
        """Validate that DNA sequences contain only A, C, G, and T."""
        with pytest.raises(ValidationError):
            DNA(sequence=sequence)

    def test_validate_modify(self: Self) -> None:
        """Validate that ``modify()`` appends residue modifications."""
        dna: DNA = DNA(sequence="GACCTCT")

        out: DNA = dna.modify(
            Modification(type="6OG", position=1),
            Modification(type="6MA", position=2),
        )

        assert out is dna
        assert len(dna.modifications) == 2
        assert dna.modifications[0].type == "6OG"
        assert dna.modifications[0].position == 1
        assert dna.modifications[1].type == "6MA"
        assert dna.modifications[1].position == 2

    def test_validate_modify_empty(self: Self) -> None:
        """Validate that ``modify()`` requires at least one modification."""
        dna: DNA = DNA(sequence="GACCTCT")

        with pytest.raises(TypeError, match="no modifications were provided"):
            dna.modify()

    def test_validate_modify_scope(self: Self) -> None:
        """Validate that DNA scope is assigned to all modifications."""
        dna: DNA = DNA(
            sequence="GACCTCT",
            modifications=(
                Modification(type="6OG", position=1),
                Modification(type="6MA", position=2),
            ),
        )

        assert all(
            modification.scope == Entity.DNA
            for modification in dna.modifications
        )

    def test_validate_modify_position(self: Self) -> None:
        """Validate that modification positions must be within the sequence."""
        with pytest.raises(ValidationError, match="out of range"):
            DNA(
                sequence="GAATTC",
                modifications=(Modification(type="6OG", position=7),),
            )

    def test_validate_modify_unique(self: Self) -> None:
        """Validate that modification positions must be unique."""
        with pytest.raises(ValidationError, match="same `position`"):
            DNA(
                sequence="GAATTC",
                modifications=(
                    Modification(type="6OG", position=2),
                    Modification(type="6MA", position=2),
                ),
            )

    @pytest.mark.parametrize("copies", [0, -1, -5])
    def test_validate_copies(self: Self, copies: int) -> None:
        """Validate that the copy count must be at least one."""
        with pytest.raises(ValidationError):
            DNA(sequence="GAATTC", copies=copies)

    def test_validate_copies_derived(self: Self) -> None:
        """Validate that the copy count is derived from multiple identifiers."""
        dna: DNA = DNA(
            sequence="GAATTC",
            id=("A", "B", "C"),
        )

        assert dna.id == ("A", "B", "C")
        assert dna.copies == 3

    def test_validate_copies_consistent(self: Self) -> None:
        """Validate consistency between explicit copies and identifiers."""
        with pytest.raises(
            ValidationError,
            match="inconsistent with the length of `id`",
        ):
            DNA(
                sequence="GAATTC",
                id=("A", "B"),
                copies=3,
            )

    def test_serialize(self: Self) -> None:
        """Validate serialization to the wrapped AlphaFold 3 form."""
        dna: DNA = DNA(
            sequence="GAATTC",
            description="EcoRI site",
        )

        data: dict[str, object] = dna.model_dump()

        assert "dna" in data

        wrapped: object = data["dna"]
        assert isinstance(wrapped, dict)
        assert wrapped["id"] is None
        assert wrapped["description"] == "EcoRI site"
        assert wrapped["sequence"] == "GAATTC"
        assert wrapped["modifications"] == ()
        assert "copies" not in wrapped
