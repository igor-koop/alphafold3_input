"""Tests for covalent bond models."""

from __future__ import annotations

from typing import Self

import pytest
from pydantic import ValidationError

from alphafold3_input.bond import Atom, Bond


class TestAtom:
    """Tests for the ``Atom`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        atom: Atom = Atom(entity="A", residue=1, name="CB")

        assert atom.entity == "A"
        assert atom.residue == 1
        assert atom.name == "CB"

    def test_construct_sequence(self: Self) -> None:
        """Validate construction from a compact sequence."""
        atom: Atom = Atom.model_validate(["A", 1, "CB"])

        assert atom.entity == "A"
        assert atom.residue == 1
        assert atom.name == "CB"

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ([], "Invalid atom definition"),
            (["A"], "Invalid atom definition"),
            (["A", 1], "Invalid atom definition"),
            (["A", 1, "CB", "extra"], "Invalid atom definition"),
        ],
    )
    def test_validate_sequence(
        self: Self,
        data: list[object],
        message: str,
    ) -> None:
        """Validate that compact input must contain exactly three items."""
        with pytest.raises(ValidationError, match=message):
            Atom.model_validate(data)

    @pytest.mark.parametrize(
        "entity",
        [
            "a",
            "Ab",
            "A1",
            "A_B",
            "A-B",
            "",
        ],
    )
    def test_validate_entity(self: Self, entity: str) -> None:
        """Validate bijective base-26 format of the entity identifier."""
        with pytest.raises(ValidationError):
            Atom(entity=entity, residue=1, name="CB")

    @pytest.mark.parametrize("residue", [0, -1, -5])
    def test_validate_residue(self: Self, residue: int) -> None:
        """Validate that the residue index must be at least one."""
        with pytest.raises(ValidationError):
            Atom(entity="A", residue=residue, name="CB")

    def test_validate_name(self: Self) -> None:
        """Validate that the atom name must be non-empty."""
        with pytest.raises(ValidationError):
            Atom(entity="A", residue=1, name="")

    def test_serialize(self: Self) -> None:
        """Validate serialization to the compact AlphaFold 3 form."""
        atom: Atom = Atom(entity="A", residue=1, name="CB")

        assert atom.model_dump() == ("A", 1, "CB")


class TestBond:
    """Tests for the ``Bond`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        bond: Bond = Bond(
            source=Atom(entity="A", residue=1, name="CA"),
            target=Atom(entity="G", residue=1, name="CHA"),
        )

        assert bond.source == Atom(entity="A", residue=1, name="CA")
        assert bond.target == Atom(entity="G", residue=1, name="CHA")

    def test_construct_sequence(self: Self) -> None:
        """Validate construction from a compact nested sequence."""
        bond: Bond = Bond.model_validate(
            [
                ["A", 1, "CA"],
                ["G", 1, "CHA"],
            ],
        )

        assert bond.source == Atom(entity="A", residue=1, name="CA")
        assert bond.target == Atom(entity="G", residue=1, name="CHA")

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            (
                [],
                "Invalid bond definition",
            ),
            (
                [["A", 1, "CA"]],
                "Invalid bond definition",
            ),
            (
                [["A", 1, "CA"], ["G", 1, "CHA"], ["X", 1, "N"]],
                "Invalid bond definition",
            ),
        ],
    )
    def test_validate_sequence(
        self: Self,
        data: list[object],
        message: str,
    ) -> None:
        """Validate that compact input must contain exactly two items."""
        with pytest.raises(ValidationError, match=message):
            Bond.model_validate(data)

    def test_validate_atom(self: Self) -> None:
        """Validate nested ``Atom`` model validation."""
        with pytest.raises(ValidationError):
            Bond.model_validate(
                [
                    ["a", 1, "CA"],
                    ["G", 1, "CHA"],
                ],
            )

    def test_serialize(self: Self) -> None:
        """Validate serialization to the compact AlphaFold 3 form."""
        bond: Bond = Bond(
            source=Atom(entity="A", residue=1, name="CA"),
            target=Atom(entity="G", residue=1, name="CHA"),
        )

        assert bond.model_dump() == (
            ("A", 1, "CA"),
            ("G", 1, "CHA"),
        )
