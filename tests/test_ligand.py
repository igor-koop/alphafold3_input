"""Tests for ligand entity models."""

from __future__ import annotations

from typing import Self

import pytest
from pydantic import ValidationError

from alphafold3_input.ligand import Ligand


class TestLigand:
    """Tests for the ``Ligand`` model."""

    def test_construct_mapping_ccd(self: Self) -> None:
        """Validate construction from a mapping with CCD code definition."""
        ligand: Ligand = Ligand(
            description="Adenosine triphosphate",
            definition=["ATP"],
        )

        assert ligand.id is None
        assert ligand.description == "Adenosine triphosphate"
        assert ligand.definition == ["ATP"]
        assert ligand.copies == 1

    def test_construct_mapping_smiles(self: Self) -> None:
        """Validate construction from a mapping with SMILES definition."""
        ligand: Ligand = Ligand(
            description="Aceclidine",
            definition="CC(=O)OC1C[NH+]2CCC1CC2",
        )

        assert ligand.id is None
        assert ligand.description == "Aceclidine"
        assert ligand.definition == "CC(=O)OC1C[NH+]2CCC1CC2"
        assert ligand.copies == 1

    def test_construct_aliases_ccd(self: Self) -> None:
        """Validate construction from wrapped CCD code aliases."""
        ligand: Ligand = Ligand.model_validate(
            {
                "ligand": {
                    "description": "Magnesium ion",
                    "ccdCodes": ["MG"],
                },
            },
        )

        assert ligand.id is None
        assert ligand.description == "Magnesium ion"
        assert ligand.definition == ["MG"]
        assert ligand.copies == 1

    def test_construct_aliases_smiles(self: Self) -> None:
        """Validate construction from wrapped SMILES aliases."""
        ligand: Ligand = Ligand.model_validate(
            {
                "ligand": {
                    "id": ["LIG"],
                    "description": "Aceclidine",
                    "smiles": "CC(=O)OC1C[NH+]2CCC1CC2",
                },
            },
        )

        assert ligand.id == ["LIG"]
        assert ligand.description == "Aceclidine"
        assert ligand.definition == "CC(=O)OC1C[NH+]2CCC1CC2"
        assert ligand.copies == 1

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
        """Validate bijective base-26 format of ligand identifiers."""
        with pytest.raises(ValidationError):
            Ligand(definition=["ATP"], id=identifier)

    @pytest.mark.parametrize("copies", [0, -1, -5])
    def test_validate_copies(self: Self, copies: int) -> None:
        """Validate that the ligand copy count must be at least one."""
        with pytest.raises(ValidationError):
            Ligand(definition=["ATP"], copies=copies)

    def test_validate_copies_derived(self: Self) -> None:
        """Validate that the copy count is derived from explicit identifiers."""
        ligand: Ligand = Ligand(
            id=["A", "B", "C"],
            definition=["MG"],
        )

        assert ligand.id == ["A", "B", "C"]
        assert ligand.copies == 3

    def test_validate_copies_consistency(self: Self) -> None:
        """Validate consistency between explicit copies and identifiers."""
        with pytest.raises(
            ValidationError,
            match="inconsistent with the length of `id`",
        ):
            Ligand(
                id=["A", "B"],
                definition=["MG"],
                copies=3,
            )

    def test_validate_definition_ccd(self: Self) -> None:
        """Validate that CCD code definitions are accepted unchanged."""
        ligand: Ligand = Ligand(definition=["ATP", "NAP"])

        assert ligand.definition == ["ATP", "NAP"]

    def test_validate_definition_smiles(self: Self) -> None:
        """Validate that canonical SMILES definitions are accepted."""
        ligand: Ligand = Ligand(
            definition="CC(=O)OC1C[NH+]2CCC1CC2",
        )

        assert ligand.definition == "CC(=O)OC1C[NH+]2CCC1CC2"

    def test_validate_definition_smiles_syntax(self: Self) -> None:
        """Validate that syntactically invalid SMILES are rejected."""
        with pytest.raises(ValidationError, match="not syntactically valid"):
            Ligand(definition="C1CC(")

    def test_validate_definition_smiles_chemistry(self: Self) -> None:
        """Validate that chemically invalid SMILES are rejected."""
        with pytest.raises(ValidationError, match="chemically valid structure"):
            Ligand(definition="c1ncocn1")

    def test_validate_definition_smiles_canonical(self: Self) -> None:
        """Validate that non-canonical SMILES are rejected."""
        with pytest.raises(ValidationError, match="not in canonical form"):
            Ligand(definition="C(C)O")

    def test_serialize_ccd(self: Self) -> None:
        """Validate serialization of CCD code definitions."""
        ligand: Ligand = Ligand(
            description="Adenosine triphosphate",
            definition=["ATP"],
        )

        data: dict[str, object] = ligand.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "ligand" in data
        assert isinstance(data["ligand"], dict)
        assert data["ligand"]["description"] == "Adenosine triphosphate"
        assert data["ligand"]["ccdCodes"] == ["ATP"]
        assert "smiles" not in data["ligand"]
        assert "id" not in data["ligand"]
        assert "copies" not in data["ligand"]

    def test_serialize_smiles(self: Self) -> None:
        """Validate serialization of SMILES definitions."""
        ligand: Ligand = Ligand(
            id=["LIG"],
            description="Aceclidine",
            definition="CC(=O)OC1C[NH+]2CCC1CC2",
        )

        data: dict[str, object] = ligand.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert "ligand" in data
        assert isinstance(data["ligand"], dict)
        assert data["ligand"]["id"] == ["LIG"]
        assert data["ligand"]["description"] == "Aceclidine"
        assert data["ligand"]["smiles"] == "CC(=O)OC1C[NH+]2CCC1CC2"
        assert "ccdCodes" not in data["ligand"]
        assert "copies" not in data["ligand"]
