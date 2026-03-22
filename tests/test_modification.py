"""Tests for residue modification models."""

from __future__ import annotations

from typing import Self, cast

import pytest
from pydantic import ValidationError

from alphafold3_input.modification import Entity, Modification


class TestEntity:
    """Tests for the ``Entity`` enum."""

    def test_validate_members(self: Self) -> None:
        """Validate the AlphaFold 3 entity type values."""
        assert Entity.PROTEIN == "protein"
        assert Entity.RNA == "rna"
        assert Entity.DNA == "dna"
        assert Entity.LIGAND == "ligand"


class TestModification:
    """Tests for the ``Modification`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        modification: Modification = Modification(
            type="5MC",
            position=4,
        )

        assert modification.scope is None
        assert modification.type == "5MC"
        assert modification.position == 4

    def test_construct_aliases_protein(self: Self) -> None:
        """Validate construction from protein modification aliases."""
        modification: Modification = Modification.model_validate(
            {
                "ptmType": "HY3",
                "ptmPosition": 1,
            },
        )

        assert modification.scope is None
        assert modification.type == "HY3"
        assert modification.position == 1

    def test_construct_aliases_base(self: Self) -> None:
        """Validate construction from nucleic-acid modification aliases."""
        modification: Modification = Modification.model_validate(
            {
                "modificationType": "5MC",
                "basePosition": 4,
            },
        )

        assert modification.scope is None
        assert modification.type == "5MC"
        assert modification.position == 4

    @pytest.mark.parametrize("position", [0, -1, -5])
    def test_validate_position(self: Self, position: int) -> None:
        """Validate that the modification position must be at least one."""
        with pytest.raises(ValidationError):
            Modification(type="5MC", position=position)

    def test_validate_scope(self: Self) -> None:
        """Validate that ligand scope is not accepted."""
        with pytest.raises(ValidationError):
            Modification(
                scope=cast("None", Entity.LIGAND),
                type="5MC",
                position=4,
            )

    def test_validate_scope_required(self: Self) -> None:
        """Validate that scope must be set before serialization."""
        modification: Modification = Modification(
            type="5MC",
            position=4,
        )

        with pytest.raises(ValueError, match=r"scope.*must be set"):
            modification.model_dump(by_alias=True, exclude_none=True)

    def test_serialize_protein(self: Self) -> None:
        """Validate serialization with protein scope."""
        modification: Modification = Modification(
            scope=Entity.PROTEIN,
            type="HY3",
            position=1,
        )

        data: dict[str, object] = modification.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert data["ptmType"] == "HY3"
        assert data["ptmPosition"] == 1
        assert "modificationType" not in data
        assert "basePosition" not in data
        assert "scope" not in data
        assert "type" not in data
        assert "position" not in data

    def test_serialize_dna(self: Self) -> None:
        """Validate serialization with DNA scope."""
        modification: Modification = Modification(
            scope=Entity.DNA,
            type="5MC",
            position=4,
        )

        data: dict[str, object] = modification.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert data["modificationType"] == "5MC"
        assert data["basePosition"] == 4
        assert "ptmType" not in data
        assert "ptmPosition" not in data
        assert "scope" not in data
        assert "type" not in data
        assert "position" not in data

    def test_serialize_rna(self: Self) -> None:
        """Validate serialization with RNA scope."""
        modification: Modification = Modification(
            scope=Entity.RNA,
            type="PSU",
            position=7,
        )

        data: dict[str, object] = modification.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert data["modificationType"] == "PSU"
        assert data["basePosition"] == 7
        assert "ptmType" not in data
        assert "ptmPosition" not in data
        assert "scope" not in data
        assert "type" not in data
        assert "position" not in data
