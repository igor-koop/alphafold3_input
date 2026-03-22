"""Tests for AlphaFold 3 job models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self

import pytest
from pydantic import ValidationError

from alphafold3_input import Modification
from alphafold3_input.bond import Atom, Bond
from alphafold3_input.dna import DNA
from alphafold3_input.job import Dialect, Job, Version
from alphafold3_input.ligand import Ligand
from alphafold3_input.protein import Protein
from alphafold3_input.rna import RNA
from alphafold3_input.template import Template

if TYPE_CHECKING:
    from pathlib import Path


class TestDialect:
    """Tests for the ``Dialect`` enum."""

    def test_validate_members(self: Self) -> None:
        """Validate the AlphaFold 3 dialect values."""
        assert Dialect.LOCAL == "alphafold3"
        assert Dialect.SERVER == "alphafoldserver"


class TestVersion:
    """Tests for the ``Version`` enum."""

    def test_validate_members(self: Self) -> None:
        """Validate the input format version values."""
        assert Version.I == 1
        assert Version.II == 2
        assert Version.III == 3
        assert Version.IV == 4


class TestJob:
    """Tests for the ``Job`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        job: Job = Job(name="example")

        assert job.name == "example"
        assert job.dialect == Dialect.LOCAL
        assert job.version == Version.IV
        assert isinstance(job.seeds, tuple)
        assert len(job.seeds) == 1
        assert job.entities == ()
        assert job.bonds is None
        assert job.ccd is None

    def test_construct_aliases(self: Self) -> None:
        """Validate construction from AlphaFold 3 field aliases."""
        job: Job = Job.model_validate(
            {
                "name": "example",
                "modelSeeds": [1, 2, 3],
                "sequences": [
                    {
                        "protein": {
                            "sequence": "ACDE",
                        },
                    },
                ],
                "bondedAtomPairs": [
                    [
                        ["A", 1, "CA"],
                        ["B", 1, "C1"],
                    ],
                ],
            },
        )

        assert job.name == "example"
        assert job.seeds == [1, 2, 3]
        assert len(job.entities) == 1
        assert isinstance(job.entities[0], Protein)
        assert job.bonds is not None
        assert len(job.bonds) == 1
        assert isinstance(job.bonds[0], Bond)

    def test_validate_add(self: Self) -> None:
        """Validate that ``add()`` appends entities and returns identifiers."""
        job: Job = Job(name="example")

        identifiers: tuple[tuple[str, ...], ...] = job.add(
            Protein(sequence="ACDE"),
            Ligand(definition=["BTN"]),
        )

        assert len(job.entities) == 2
        assert isinstance(job.entities[0], Protein)
        assert isinstance(job.entities[1], Ligand)
        assert identifiers == (("A",), ("B",))

    def test_validate_add_empty(self: Self) -> None:
        """Validate that ``add()`` requires at least one entity."""
        job: Job = Job(name="example")

        with pytest.raises(TypeError, match="no entities were provided"):
            job.add()

    def test_validate_entities_allocated(self: Self) -> None:
        """Validate that entity identifiers are allocated automatically."""
        job: Job = Job(
            name="example",
            entities=(
                Protein(sequence="ACDE", id=("A",)),
                DNA(sequence="GAATTC", copies=2),
                RNA(sequence="UAGCUAGC", id=("X", "Y", "Z")),
                Ligand(definition=["BTN"]),
            ),
        )

        assert job.entities[0].id == ("A",)
        assert job.entities[1].id == ("B", "C")
        assert job.entities[2].id == ("X", "Y", "Z")
        assert job.entities[3].id == ("D",)

    def test_validate_entities_duplicate(self: Self) -> None:
        """Validate that entity identifiers must be unique across the job."""
        with pytest.raises(ValidationError, match="used more than once"):
            Job(
                name="example",
                entities=(
                    Protein(sequence="ACDE", id="A"),
                    DNA(sequence="GAATTC", id="A"),
                ),
            )

    def test_validate_dialect(self: Self) -> None:
        """Validate that the server-side dialect is not supported."""
        with pytest.raises(NotImplementedError, match="not supported"):
            Job(name="example", dialect=Dialect.SERVER)

    def test_validate_version_description(self: Self) -> None:
        """Validate that entity descriptions require version four."""
        with pytest.raises(ValidationError, match=r"version.*too low"):
            Job(
                name="example",
                version=Version.III,
                entities=(
                    Protein(
                        sequence="ACDE",
                        description="example",
                    ),
                ),
            )

    def test_validate_version_ccd(self: Self, tmp_path: Path) -> None:
        """Validate that path-based CCD input requires version three."""
        path: Path = tmp_path / "components.cif"

        with pytest.raises(ValidationError, match=r"version.*too low"):
            Job(
                name="example",
                version=Version.II,
                ccd=path,
            )

    def test_validate_version_alignment(self: Self, tmp_path: Path) -> None:
        """Validate that path-based alignments require version two."""
        alignment: Path = tmp_path / "alignment.a3m"

        with pytest.raises(ValidationError, match=r"version.*too low"):
            Job(
                name="example",
                version=Version.I,
                entities=(
                    Protein(
                        sequence="ACDE",
                        alignment=alignment,
                    ),
                ),
            )

        with pytest.raises(ValidationError, match=r"version.*too low"):
            Job(
                name="example",
                version=Version.I,
                entities=(
                    RNA(
                        sequence="ACGU",
                        alignment=alignment,
                    ),
                ),
            )

    def test_validate_version_template(self: Self, tmp_path: Path) -> None:
        """Validate that path-based template structures require version two."""
        structure: Path = tmp_path / "template.cif"

        with pytest.raises(ValidationError, match=r"version.*too low"):
            Job(
                name="example",
                version=Version.I,
                entities=(
                    Protein(
                        sequence="ACDE",
                        templates=(
                            Template(
                                structure=structure,
                                indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
                            ),
                        ),
                    ),
                ),
            )

    def test_validate_bonds(self: Self) -> None:
        """Validate job construction with bonded atom pairs."""
        job: Job = Job(
            name="example",
            entities=(
                Protein(sequence="ACDE", id="A"),
                Ligand(definition=["BTN"], id="B"),
            ),
            bonds=(
                Bond(
                    source=Atom(entity="B", residue=1, name="C11"),
                    target=Atom(entity="A", residue=1, name="CA"),
                ),
            ),
        )

        assert job.bonds is not None
        assert len(job.bonds) == 1
        assert job.bonds[0].source == Atom(entity="B", residue=1, name="C11")
        assert job.bonds[0].target == Atom(entity="A", residue=1, name="CA")

    def test_validate_ccd_inline(self: Self) -> None:
        """Validate inline CCD serialization."""
        job: Job = Job(
            name="example",
            ccd="data_example",
        )

        data: dict[str, object] = job.export()

        assert data["userCCD"] == "data_example"
        assert "userCCDPath" not in data

    def test_validate_ccd_path(self: Self, tmp_path: Path) -> None:
        """Validate path-based CCD coercion and serialization."""
        path: Path = tmp_path / "components.cif"

        job: Job = Job.model_validate(
            {
                "name": "example",
                "userCCDPath": str(path),
            },
        )

        assert job.ccd == path

        data: dict[str, object] = job.export()

        assert data["userCCDPath"] == path
        assert "userCCD" not in data

    def test_validate_seeds(self: Self) -> None:
        """Validate explicit model seeds."""
        job: Job = Job(
            name="example",
            seeds=(1, 2, 3),
        )

        assert job.seeds == (1, 2, 3)

    def test_serialize_seeds(self: Self) -> None:
        """Validate seed-count expansion during serialization."""
        job: Job = Job(
            name="example",
            seeds=3,
        )

        data: dict[str, object] = job.export()

        seeds: object = data["modelSeeds"]
        assert isinstance(seeds, tuple)
        assert len(seeds) == 3
        assert all(isinstance(seed, int) for seed in seeds)
        assert all(1 <= seed <= (1 << 32) - 1 for seed in seeds)

    def test_export(self: Self) -> None:
        """Validate export to an AlphaFold 3 input mapping."""
        job: Job = Job(
            name="example",
            entities=(Protein(sequence="ACDE"),),
        )

        data: dict[str, object] = job.export()

        assert data["name"] == "example"
        assert data["dialect"] == Dialect.LOCAL
        assert data["version"] == Version.IV
        assert "modelSeeds" in data
        assert "sequences" in data
        assert "bondedAtomPairs" not in data
        assert "userCCD" not in data
        assert "userCCDPath" not in data

    def test_save(self: Self, tmp_path: Path) -> None:
        """Validate saving a job to an AlphaFold 3 input JSON."""
        path: Path = tmp_path / "job.json"
        job: Job = Job(
            name="example",
            seeds=[1, 2],
            entities=[
                Protein(
                    id=["A"],
                    sequence="ACDE",
                    modifications=[Modification(type="MCS", position=2)],
                ),
                Ligand(
                    id=["B"],
                    definition=["BTN"],
                ),
            ],
        )

        job.save(path)
        data: dict = json.loads(path.read_text())

        assert job.name == data["name"]
        assert job.dialect == data["dialect"]
        assert job.version == data["version"]
        assert job.seeds == data["modelSeeds"]
        assert len(job.entities) == len(data["sequences"])
        assert job.entities[0] == Protein(**data["sequences"][0])

    def test_load(self: Self, tmp_path: Path) -> None:
        """Validate loading a job from an AlphaFold 3 input JSON."""
        path: Path = tmp_path / "job.json"
        job: Job = Job(
            name="example",
            seeds=[1, 2],
            entities=[
                Protein(
                    id=["A"],
                    sequence="ACDE",
                    modifications=[Modification(type="MCS", position=2)],
                ),
                Ligand(
                    id=["B"],
                    definition=["BTN"],
                ),
            ],
        )

        job.save(path)
        data: Job = Job.load(path)

        assert data.name == job.name
        assert data.dialect == job.dialect
        assert data.version == job.version
        assert data.seeds == job.seeds
        assert data.entities == job.entities
