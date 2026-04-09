"""Tests for shared utility functions and enums."""

from __future__ import annotations

from typing import Self

import pytest
from rdkit import Chem

from alphafold3_input.template import Template
from alphafold3_input.utils import (
    Operation,
    base26_decoder,
    base26_encoder,
    ccd,
    component,
    realign,
    reindex,
    trace,
)


class TestOperation:
    """Tests for the ``Operation`` enum."""

    def test_validate_members(self: Self) -> None:
        """Validate the alignment operation values."""
        assert Operation.REF == "match"
        assert Operation.SUB == "substitution"
        assert Operation.INS == "insertion"
        assert Operation.DEL == "deletion"


class TestTrace:
    """Tests for ``trace()``."""

    def test_validate_reference_match(self: Self) -> None:
        """Validate an exact residue match trace."""
        operations: tuple[Operation, ...] = trace(
            reference="ACGT",
            query="ACGT",
        )

        assert operations == (
            Operation.REF,
            Operation.REF,
            Operation.REF,
            Operation.REF,
        )

    def test_validate_substitution(self: Self) -> None:
        """Validate substitution operations."""
        operations: tuple[Operation, ...] = trace(
            reference="ACGT",
            query="AGGT",
        )

        assert operations == (
            Operation.REF,
            Operation.SUB,
            Operation.REF,
            Operation.REF,
        )

    def test_validate_deletion(self: Self) -> None:
        """Validate deletion operations."""
        operations: tuple[Operation, ...] = trace(
            reference="ACGT",
            query="A-GT",
        )

        assert operations == (
            Operation.REF,
            Operation.DEL,
            Operation.REF,
            Operation.REF,
        )

    def test_validate_insertion(self: Self) -> None:
        """Validate insertion operations."""
        operations: tuple[Operation, ...] = trace(
            reference="ACGT",
            query="AcCGT",
        )

        assert operations == (
            Operation.REF,
            Operation.INS,
            Operation.REF,
            Operation.REF,
            Operation.REF,
        )

    def test_validate_reference(self: Self) -> None:
        """Validate reference FASTA format checking."""
        with pytest.raises(
            ValueError,
            match="reference sequence",
        ):
            trace(
                reference="ACGX",
                query="ACGT",
            )

    def test_validate_query(self: Self) -> None:
        """Validate query A3M format checking."""
        with pytest.raises(
            ValueError,
            match="query sequence",
        ):
            trace(
                reference="ACGT",
                query="ACG!",
            )


class TestReindex:
    """Tests for ``reindex()``."""

    def test_validate_reference_match(self: Self) -> None:
        """Validate template reindexing for exact matches."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 0, 1: 1, 2: 2},
        )

        updated: Template = reindex(
            template=template,
            operations=(
                Operation.REF,
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated.indexes == {0: 0, 1: 1, 2: 2}

    def test_validate_deletion(self: Self) -> None:
        """Validate template reindexing across deletions."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 10, 1: 11, 2: 12, 3: 13},
        )

        updated: Template = reindex(
            template=template,
            operations=(
                Operation.REF,
                Operation.DEL,
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated.indexes == {0: 10, 1: 12, 2: 13}

    def test_validate_insertion(self: Self) -> None:
        """Validate template reindexing across insertions."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 10, 1: 11, 2: 12},
        )

        updated: Template = reindex(
            template=template,
            operations=(
                Operation.REF,
                Operation.INS,
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated.indexes == {0: 10, 2: 11, 3: 12}

    def test_validate_operation(self: Self) -> None:
        """Validate that unexpected reindex operations are rejected."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 0},
        )

        with pytest.raises(ValueError, match="Unexpected alignment operation"):
            reindex(template=template, operations=("weird",))  # ty:ignore[invalid-argument-type]

    def test_validate_missing_positions(self: Self) -> None:
        """Validate that unmapped reference positions are dropped."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 10, 1: 11, 4: 14},
        )

        updated: Template = reindex(
            template=template,
            operations=(
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated.indexes == {0: 10, 1: 11}


class TestRealign:
    """Tests for ``realign()``."""

    def test_validate_reference_match(self: Self) -> None:
        """Validate alignment realignment for exact matches."""
        alignment: str = ">seq1\nACGT\n>seq2\nAGGT"
        updated: str = realign(
            alignment=alignment,
            operations=(
                Operation.REF,
                Operation.REF,
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated == alignment

    def test_validate_insertion(self: Self) -> None:
        """Validate alignment realignment across insertions."""
        alignment: str = ">seq1\nACGT\n"
        updated: str = realign(
            alignment=alignment,
            operations=(
                Operation.REF,
                Operation.INS,
                Operation.REF,
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated == ">seq1\nA-CGT"

    def test_validate_deletion(self: Self) -> None:
        """Validate alignment realignment across deletions."""
        alignment: str = ">seq1\nACGT\n"
        updated: str = realign(
            alignment=alignment,
            operations=(
                Operation.REF,
                Operation.DEL,
                Operation.REF,
                Operation.REF,
            ),
        )

        assert updated == ">seq1\nAcGT"

    def test_validate_header(self: Self) -> None:
        """Validate A3M header checking."""
        with pytest.raises(ValueError, match="header format"):
            realign(
                alignment="seq1\nACGT",
                operations=(Operation.REF, Operation.REF),
            )

    def test_validate_sequence(self: Self) -> None:
        """Validate A3M sequence checking."""
        with pytest.raises(ValueError, match="sequence format"):
            realign(
                alignment=">seq1\nACG!",
                operations=(Operation.REF, Operation.REF),
            )

    def test_validate_short(self: Self) -> None:
        """Validate detection of sequences shorter than the trace."""
        with pytest.raises(ValueError, match="ended too early"):
            realign(
                alignment=">seq1\nAC",
                operations=(
                    Operation.REF,
                    Operation.REF,
                    Operation.REF,
                ),
            )

    def test_validate_long(self: Self) -> None:
        """Validate detection of sequences longer than the trace."""
        with pytest.raises(ValueError, match="longer than operations trace"):
            realign(
                alignment=">seq1\nACGT",
                operations=(
                    Operation.REF,
                    Operation.REF,
                ),
            )

    def test_validate_trailing_insertions(self: Self) -> None:
        """Validate detection of trailing insertions."""
        with pytest.raises(ValueError, match="trailing insertion"):
            realign(
                alignment=">seq1\nAc",
                operations=(Operation.REF, Operation.DEL),
            )


class TestComponent:
    """Tests for ``component()``."""

    @pytest.mark.parametrize(
        ("smiles", "code", "name"),
        [
            ("CCO", "ETOH", "Ethanol"),
            ("c1ccccc1", "BNZ", "Benzene"),
            ("C#CC(=O)O", "PROP", "Propiolic acid"),
            (r"C/C=C/C=C\C", "EZHEX", "(2E,4Z)-hexa-2,4-diene"),
            ("C1C[C@]2(Cl)CC[C@@]1(Cl)C2", "DCLNOR", "1,4-dichloronorbornane"),
            ("C([C@@H](C(=O)O)N)S", "CYS", "Cysteine"),
        ],
    )
    def test_validate_export(
        self: Self,
        smiles: str,
        code: str,
        name: str,
    ) -> None:
        """Validate generation of an annotated embedded component."""
        molecule: Chem.Mol = component(smiles=smiles, code=code, name=name)

        assert molecule.GetProp("comp_id") == code
        assert molecule.GetProp("comp_name") == name
        assert molecule.GetProp("comp_smiles") == smiles
        assert molecule.GetNumConformers() == 1

        atom_names: list[str] = [
            atom.GetProp("atom_name") for atom in molecule.GetAtoms()
        ]
        assert atom_names
        assert all(atom_name for atom_name in atom_names)

    def test_validate_code(self: Self) -> None:
        """Validate chemical component code format checking."""
        with pytest.raises(ValueError, match="component code"):
            component(
                smiles="CCO",
                code="etoh",
                name="Ethanol",
            )

    @pytest.mark.parametrize(
        "smiles",
        ["-c1ccccc1c", "c1c(cccc1", "c1cccc"],
    )
    def test_validate_smiles_syntax(self: Self, smiles: str) -> None:
        """Validate syntactic SMILES checking."""
        with pytest.raises(ValueError, match="syntactically valid"):
            component(
                smiles=smiles,
                code="BAD",
                name="Benzene",
            )

    def test_validate_smiles_chemistry(self: Self) -> None:
        """Validate chemical SMILES checking."""
        with pytest.raises(ValueError, match="chemically valid structure"):
            component(
                smiles="c1ncocn1",
                code="BAD",
                name="Unsubstituted 1,3,4-oxadiazine",
            )

    def test_validate_embedding(self: Self) -> None:
        """Validate rejection when conformer embedding fails."""
        with pytest.raises(RuntimeError, match="conformer generation"):
            component(
                smiles="C1C[C@]2(Cl)CC[C@]1(Cl)C2",
                code="BAD",
                name="Strained 1,4-dichloronorbornane",
            )


class TestCCD:
    """Tests for ``ccd()``."""

    @pytest.mark.parametrize(
        ("smiles", "code", "name"),
        [
            ("CCO", "ETOH", "Ethanol"),
            ("c1ccccc1", "BNZ", "Benzene"),
            ("C#CC(=O)O", "PROP", "Propiolic acid"),
            (r"C/C=C/C=C\C", "EZHEX", "(2E,4Z)-hexa-2,4-diene"),
            ("C1C[C@]2(Cl)CC[C@@]1(Cl)C2", "DCLNOR", "1,4-dichloronorbornane"),
            ("C([C@@H](C(=O)O)N)S", "CYS", "Cysteine"),
        ],
    )
    def test_validate_export(
        self: Self,
        smiles: str,
        code: str,
        name: str,
    ) -> None:
        """Validate CCD export for supported components."""
        molecule: Chem.Mol = component(smiles=smiles, code=code, name=name)

        blocks: list[str] = list(ccd(molecule))

        assert len(blocks) == 1
        assert f"data_{code}" in blocks[0]
        assert f"_chem_comp.id {code}" in blocks[0]
        assert f"_chem_comp.name {name}" in blocks[0]
        assert f"_chem_comp.pdbx_smiles {smiles}" in blocks[0]
        assert "_chem_comp_atom.atom_id" in blocks[0]
        assert "_chem_comp_bond.atom_id_1" in blocks[0]

    def test_validate_property(self: Self) -> None:
        """Validate required molecule property checking."""
        molecule = component(
            smiles="CCO",
            code="ETOH",
            name="Ethanol",
        )

        molecule.ClearProp("comp_name")

        with pytest.raises(KeyError, match="missing required property"):
            list(ccd(molecule))

    def test_validate_conformer(self: Self) -> None:
        """Validate rejection of components without conformers."""
        molecule: Chem.Mol = component(
            smiles="CCO",
            code="ETOH",
            name="Ethanol",
        )

        molecule.RemoveAllConformers()

        with pytest.raises(ValueError, match="no available conformers"):
            list(ccd(molecule))

    def test_validate_coordinates(self: Self) -> None:
        """Validate rejection of components with unavailable coordinates."""
        molecule: Chem.Mol = component(
            smiles="CCO",
            code="ETOH",
            name="Ethanol",
        )

        class DummyConformer:
            def GetPositions() -> None:  # noqa: N802
                return None

        setattr(molecule, "GetConformer", lambda _: DummyConformer)  # noqa: B010

        with pytest.raises(ValueError, match="coordinates are unavailable"):
            list(ccd(molecule))

    def test_validate_bond_type(self: Self) -> None:
        """Validate rejection of unsupported bond types."""
        molecule: Chem.Mol = component(
            smiles="[Na+].[Cl-]",
            code="NACL",
            name="Sodium chloride",
        )

        molecule = Chem.RWMol(molecule)
        molecule.AddBond(0, 1, Chem.BondType.IONIC)

        with pytest.raises(TypeError, match="unsupported bond type"):
            list(ccd(molecule))

    def test_validate_bond_stereo(self: Self) -> None:
        """Validate rejection of unsupported bond stereochemistry."""
        molecule: Chem.Mol = component(
            smiles="CC=CC |w:1.0|",
            code="2BUT",
            name="(E/Z)-but-2-ene",
        )

        with pytest.raises(TypeError, match="unsupported bond stereochemistry"):
            list(ccd(molecule))


class TestBase26:
    """Tests for bijective base-26 helpers."""

    @pytest.mark.parametrize(
        ("value", "label"),
        [
            (1, "A"),
            (26, "Z"),
            (27, "AA"),
            (52, "AZ"),
            (53, "BA"),
            (702, "ZZ"),
            (703, "AAA"),
        ],
    )
    def test_validate_encoder(
        self: Self,
        value: int,
        label: str,
    ) -> None:
        """Validate bijective base-26 encoding."""
        assert base26_encoder(value) == label

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("A", 1),
            ("Z", 26),
            ("AA", 27),
            ("AZ", 52),
            ("BA", 53),
            ("ZZ", 702),
            ("AAA", 703),
        ],
    )
    def test_validate_decoder(
        self: Self,
        label: str,
        value: int,
    ) -> None:
        """Validate bijective base-26 decoding."""
        assert base26_decoder(label) == value

    def test_validate_roundtrip(self: Self) -> None:
        """Validate roundtrip base-26 conversion."""
        for value in (1, 2, 26, 27, 52, 53, 702, 703, 18278):
            assert base26_decoder(base26_encoder(value)) == value

    @pytest.mark.parametrize("value", [0, -1, -5])
    def test_validate_encoder_input(self: Self, value: int) -> None:
        """Validate that encoded integers must be positive."""
        with pytest.raises(ValueError, match="positive 1-based integer"):
            base26_encoder(value)

    @pytest.mark.parametrize(
        "label",
        [
            "",
            "a",
            "A1",
            "A-",
            "A_",
        ],
    )
    def test_validate_decoder_input(self: Self, label: str) -> None:
        """Validate that decoded labels must contain only uppercase letters."""
        with pytest.raises(ValueError, match="uppercase letters"):
            base26_decoder(label)
