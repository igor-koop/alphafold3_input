"""AlphaFold 3 job model.

This submodule defines the `Job` model, which represents a complete AlphaFold 3
input configuration for local execution. The module also provides `Dialect` and
`Version` enums to control input file format.

Exports:
    Dialect: Enum selecting the AlphaFold input dialect.
    Version: Enum selecting the AlphaFold 3 input format version.
    Job: Model representing a complete AlphaFold 3 job input.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    SerializerFunctionWrapHandler,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from .bond import Bond
from .dna import DNA
from .ligand import Ligand
from .protein import Protein
from .rna import RNA
from .utils import base26_decoder, base26_encoder

__all__: list[str] = [
    "Dialect",
    "Job",
    "Version",
]


class Dialect(StrEnum):
    """AlphaFold 3 input format dialect.

    Members:
        LOCAL: AlphaFold 3 dialect (`"alphafold3"`).
        SERVER: AlphaFoldServer dialect (`"alphafoldserver"`).

    """

    LOCAL = "alphafold3"
    SERVER = "alphafoldserver"


class Version(IntEnum):
    """AlphaFold 3 input format version.

    Members:
        I: Input format version 1.
        II: Input format version 2.
        III: Input format version 3.
        IV: Input format version 4.
    """

    I = 1  # noqa: E741
    II = 2
    III = 3
    IV = 4


class Job(BaseModel):
    """AlphaFold 3 job specification.

    Represents a single AlphaFold 3 input job containing one or more sequence
    entities (`Protein`, `RNA`, `DNA`, or `Ligand`). Entities can be appended
    incrementally via `Job.add()`.

    The number of predicted structures is controlled by `seeds`, which can be
    specified either as an integer count or as an explicit sequence of integer
    seeds.

    A job may also include explicit covalent `bonds` (bonded atom pairs)
    between atoms within a ligand or between a ligand and a polymeric entity.
    A job may also provide a custom `ccd` (chemical components dictionary)
    either inline as a string or via a filesystem path.

    The selected input `version` must support the features used by the job. The
    `dialect` selects the AlphaFold 3 input format and currently only supports
    `Dialect.LOCAL` (`alphafold3`).

    Attributes:
        name (str): Job name.
        dialect (Dialect): Input dialect.
        version (Version): Input format version.
        seeds (int | Sequence[int]): Random seeds or their total number.
        entities (Sequence[Protein | RNA | DNA | Ligand]): Entities included in
            the job.
        bonds (Sequence[Bond]): Covalent bonds between atom pairs.
        ccd (str | Path | None): Custom chemical components dictionary.

    Examples:
        Job with a protein and a covalently-linked ligand.
        ```python
        job = Job(name="example")

        ((carboxylase,), (biotin,)) = job.add(
            Protein(sequence="VLSAMKMETVV"),
            Ligand(definition=["BTN"]),
        )

        job.bonds = (
            Bond(
                source=Atom(entity=biotin, residue=1, name="C11"),
                target=Atom(entity=carboxylase, residue=6, name="NZ"),
            ),
        )
        ```

        Job with multiple entity copies and multiple model seeds.
        ```python
        Job(
            name="multimer",
            seeds=5,
            entities=[
                Protein(
                    sequence="ACDE",
                    description="homotrimer",
                    copies=3,
                ),
            ],
        )
        ```

    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
        use_enum_values=False,
    )

    name: Annotated[
        str,
        Field(
            title="name",
            description="Job name.",
            validation_alias="name",
            serialization_alias="name",
        ),
    ]

    dialect: Annotated[
        Dialect,
        Field(
            title="dialect",
            description="Input dialect.",
            validation_alias="dialect",
            serialization_alias="dialect",
        ),
    ] = Dialect.LOCAL

    version: Annotated[
        Version,
        Field(
            title="version",
            description="Input format version.",
            validation_alias="version",
            serialization_alias="version",
        ),
    ] = Version.IV

    seeds: Annotated[
        Annotated[int, Field(ge=1)]
        | Annotated[
            Sequence[Annotated[int, Field(ge=1, le=(1 << 32) - 1)]],
            Field(min_length=1),
        ],
        Field(
            title="seeds",
            description="Random seeds or their total number.",
            validation_alias="modelSeeds",
            serialization_alias="modelSeeds",
        ),
    ] = Field(default_factory=lambda: (random.getrandbits(32) or 1,))

    entities: Annotated[
        Sequence[Protein | RNA | DNA | Ligand],
        Field(
            title="entities",
            description="Entities included in the job.",
            validation_alias="sequences",
            serialization_alias="sequences",
        ),
    ] = Field(default_factory=tuple)

    bonds: Annotated[
        Sequence[Bond],
        Field(
            title="bonds",
            description="Covalent bonds between atom pairs.",
            validation_alias="bondedAtomPairs",
            serialization_alias="bondedAtomPairs",
        ),
    ] = Field(default_factory=tuple)

    ccd: Annotated[
        str | Path | None,
        Field(
            title="ccd",
            description="Custom chemical components dictionary.",
            validation_alias=AliasChoices("userCCD", "userCCDPath"),
            exclude=True,
        ),
    ] = None

    @computed_field(alias="userCCD", repr=False)
    @property
    def __ccd_inline(self) -> str | None:
        """Expose inline `ccd` for serialization.

        Returns:
            out (str | None): `ccd` when it is a string, otherwise `None`.

        """
        return self.ccd if isinstance(self.ccd, str) else None

    @computed_field(alias="userCCDPath", repr=False)
    @property
    def __ccd_path(self) -> Path | None:
        """Expose `ccd` path for serialization.

        Returns:
            out (Path | None): `ccd` when it is a path, otherwise `None`.

        """
        return self.ccd if isinstance(self.ccd, Path) else None

    @staticmethod
    def __allocate_ids(
        entities: Sequence[Protein | RNA | DNA | Ligand],
        ids: set[int],
    ) -> None:
        """Assign missing entity identifiers in-place.

        Allocates identifiers for each entity whose `id` is unset (`None`).
        For entities with `copies > 1`, a distinct identifier is assigned per
        copy.

        Args:
            entities (Sequence[Protein | RNA | DNA | Ligand]): Entities that
                may require identifier assignment.
            ids (set[int]): Identifiers reserved across the job (numeric form).

        """
        candidate: int = 1

        for entity in entities:
            if entity.id is not None:
                continue

            allocated: list[int] = []
            while len(allocated) < entity.copies:
                if candidate in ids:
                    candidate += 1
                    continue

                allocated.append(candidate)
                ids.add(candidate)
                candidate += 1

            entity.id = tuple(base26_encoder(i) for i in allocated)

    def add(
        self: Self,
        *entities: Protein | RNA | DNA | Ligand,
    ) -> tuple[tuple[str, ...], ...]:
        """Append one or more entities to an AlphaFold 3 job.

        Args:
            *entities (Protein | RNA | DNA | Ligand): One or more entities to
                add.

        Returns:
            out (tuple[tuple[str, ...], ...]): Identifiers of all added
                entities.

        Raises:
            TypeError: If no entities were provided.

        """
        if not entities:
            msg: str = "Invalid job entity: no entities were provided."
            raise TypeError(msg)

        self.entities = (*tuple(self.entities), *entities)

        return tuple(
            tuple(entity.id)
            for entity in self.entities[-len(entities) :]
            if entity.id
        )

    @classmethod
    def load(
        cls: type[Self],
        path: Path,
        *,
        encoding: str = "utf-8",
    ) -> Self:
        """Load a `Job` from an AlphaFold 3 input file.

        Reads the file at `path` as text and validates it against the `Job`
        model using Pydantic's JSON parsing and validation.

        Args:
            path (Path): Path to the JSON input file.
            encoding (str): Text encoding used to read the file.

        Returns:
            out (Self): Parsed and validated `Job` instance.

        """
        return cls.model_validate_json(
            Path(path).read_text(encoding=encoding),
        )

    def export(self: Self) -> dict[str, object]:
        """Export the `Job` to an AlphaFold 3 input mapping.

        Serializes the `Job` as a mapping for AlphaFold 3 input.

        Returns:
            out (dict[str, object]): AlphaFold 3 input mapping.

        """
        return self.model_dump(
            by_alias=True,
            exclude_none=True,
        )

    def save(
        self: Self,
        path: Path,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        encoding: str = "utf-8",
    ) -> Path:
        """Save the `Job` to an AlphaFold 3 input file.

        Serializes the `Job` to `path` as a JSON file for AlphaFold 3 input.

        Args:
            path (Path): Destination path for the JSON file.
            indent (int | None): JSON indentation level.
            ensure_ascii (bool): Whether to escape non-ASCII characters in the
                JSON output.
            encoding (str): Text encoding used to write the file.

        Returns:
            out (Path): The resolved path that was written.

        """
        file = Path(path)
        file.write_text(
            self.model_dump_json(
                by_alias=True,
                exclude_none=True,
                indent=indent,
                ensure_ascii=ensure_ascii,
            ),
            encoding=encoding,
        )
        return file

    @model_validator(mode="wrap")
    @classmethod
    def __coerce_ccd(
        cls: type[Self],
        data: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Coerce `ccd` to a `Path`.

        Inspects the raw input data. When the input is a mapping containing
        `userCCDPath`, its value is converted to a `Path` and assigned to
        `ccd` after successful model validation.

        Args:
            data (Any): Raw input data.
            handler (ModelWrapValidatorHandler[Self]): Inner model validator.

        Returns:
            out (Self): Validated model with `ccd` coerced to `Path` when
                applicable.

        """
        if not isinstance(data, Mapping):
            return handler(data)

        alias: str = "userCCDPath"
        value: object | None = data.get(alias)
        model: Self = handler(data)
        if value is not None and isinstance(value, str):
            object.__setattr__(model, "ccd", Path(value))
        return model

    @field_validator("entities", mode="after")
    @classmethod
    def __validate_entities(
        cls: type[Self],
        entities: Sequence[Protein | RNA | DNA | Ligand],
    ) -> Sequence[Protein | RNA | DNA | Ligand]:
        """Validate entity identifiers.

        Ensures that any explicitly provided entity identifiers are unique
        across the job and assigns identifiers to entities where `id` is unset
        (`None`).

        Args:
            entities (Sequence[Protein | RNA | DNA | Ligand]): Job entities.

        Returns:
            out (Sequence[Protein | RNA | DNA | Ligand]): Validated job
                entities.

        Raises:
            ValueError: If the same identifier is used more than once.

        """
        ids: set[int] = set()

        for entity in entities:
            if entity.id is None:
                continue

            for eid in entity.id:
                n: int = base26_decoder(eid)
                if n in ids:
                    msg: str = (
                        "Conflicting job configuration: entity identifier is "
                        f"used more than once (id={eid})."
                    )
                    raise ValueError(msg)
                ids.add(n)

        cls.__allocate_ids(entities, ids)
        return entities

    @field_validator("dialect", mode="after")
    @classmethod
    def __validate_dialect(cls: type[Self], value: Dialect) -> Dialect:
        """Validate the selected AlphaFold 3 input dialect.

        Args:
            value (Dialect): Requested input dialect.

        Returns:
            out (Dialect): The validated dialect.

        Raises:
            NotImplementedError: If the server-side dialect is selected.

        """
        if value == Dialect.SERVER:
            msg = f"'{Dialect.SERVER}' dialect is not supported."
            raise NotImplementedError(msg)
        return value

    @model_validator(mode="after")
    def __validate_version(self: Self) -> Self:
        """Validate that `version` supports the used input features.

        Inspects the job configuration and computes the minimum required input
        format version. Path-based `alignment` and `Template.structure` require
        `Version.II`. User-defined `ccd` requires `Version.III`. Any entity
        `description` requires `Version.IV`.

        Returns:
            out (Self): The validated job instance.

        Raises:
            ValueError: If `version` is lower than the minimum required version
                for the used features.

        """
        required: Version = Version.I

        if isinstance(self.ccd, Path):
            required = max(required, Version.III)

        for entity in self.entities:
            if entity.description is not None:
                required = max(required, Version.IV)
                continue

            if isinstance(entity, RNA) and isinstance(entity.alignment, Path):
                required = max(required, Version.II)

            if isinstance(entity, Protein):
                if isinstance(entity.alignment, Path):
                    required = max(required, Version.II)
                if any(
                    isinstance(template.structure, Path)
                    for template in entity.templates
                ):
                    required = max(required, Version.II)

        if self.version < required:
            msg: str = (
                "Invalid job configuration: `version` is too low for the "
                f"selected features (version={self.version}, "
                f"required>={required})."
            )
            raise ValueError(msg)

        return self

    @field_serializer("number", mode="wrap")
    def __expand_seeds(
        self: Self,
        value: int | Sequence[int],
        handler: SerializerFunctionWrapHandler,
    ) -> Sequence[int]:
        """Expand `seeds` into a tuple of explicit positive 32-bit seeds.

        When `seeds` is an integer count, it is expanded at serialization time
        into a tuple of `n` pseudo-random 32-bit seeds. When `seeds` is already
        a sequence, it is forwarded unchanged.

        Args:
            value (int | Sequence[int]): Random seeds or their total number.
            handler (SerializerFunctionWrapHandler): Pydantic serializer
                wrapper.

        Returns:
            Sequence[int]: Model seeds.

        """
        if isinstance(value, int):
            value = tuple(random.getrandbits(32) or 1 for _ in range(value))
        return handler(value)
