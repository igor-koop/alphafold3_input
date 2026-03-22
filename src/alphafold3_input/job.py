"""AlphaFold 3 job models.

This submodule defines :class:`Job`, which represents a complete AlphaFold 3
input configuration for local execution. It also provides :class:`Dialect`
and :class:`Version` enums for selecting the input format.

Exports:
    - :class:`Dialect`: AlphaFold input dialect enum.
    - :class:`Version`: AlphaFold 3 input format version enum.
    - :class:`Job`: Complete AlphaFold 3 job input model.
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
    """AlphaFold 3 input format dialect."""

    LOCAL = "alphafold3"
    """AlphaFold 3 dialect"""

    SERVER = "alphafoldserver"
    """AlphaFoldServer dialect."""


class Version(IntEnum):
    """AlphaFold 3 input format version."""

    I = 1  # noqa: E741
    """Input format version 1."""

    II = 2
    """Input format version 2."""

    III = 3
    """Input format version 3."""

    IV = 4
    """Input format version 4."""


class Job(BaseModel):
    """AlphaFold 3 job specification.

    A job contains one or more sequence entities (:class:`Protein`,
    :class:`RNA`, :class:`DNA`, or :class:`Ligand`) and may include explicit
    covalent :attr:`bonds` and a custom :attr:`ccd`.

    The number of predicted structures is controlled by :attr:`seeds`, which may
    be given either as an integer count or as an explicit sequence of integer
    seeds.

    The selected :attr:`version` must support the features used by the job. The
    :attr:`dialect` selects the AlphaFold 3 input format and currently only
    supports :attr:`Dialect.LOCAL`.

    Attributes:
        name (str): Job name.
        dialect (Dialect): Input format dialect.
        version (Version): Input format version.
        seeds (int | Sequence[int]): Random seeds or their total number.
        entities (Sequence[Protein | RNA | DNA | Ligand]): Entities
            included in the job.
        bonds (Sequence[Bond] | None): Covalent bonds between atom pairs.
        ccd (str | Path | None): Custom chemical components dictionary.

    Examples:
        Job with a protein and a covalently linked ligand.

        >>> job = Job(name="example")
        >>> ((carboxylase,), (biotin,)) = job.add(
        ...     Protein(sequence="VLSAMKMETVV"),
        ...     Ligand(definition=["BTN"]),
        ... )
        >>> job.bonds = (
        ...     Bond(
        ...         source=Atom(entity=biotin, residue=1, name="C11"),
        ...         target=Atom(entity=carboxylase, residue=6, name="NZ"),
        ...     ),
        ... )

        Job with multiple entity copies and multiple model seeds.

        >>> Job(
        ...     name="multimer",
        ...     seeds=5,
        ...     entities=[
        ...         Protein(
        ...             sequence="ACDE",
        ...             description="homotrimer",
        ...             copies=3,
        ...         ),
        ...     ],
        ... )
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
        use_enum_values=False,
    )

    name: str = Field(
        title="name",
        alias="name",
        description="Job name.",
        validation_alias="name",
        serialization_alias="name",
    )
    """Job name."""

    dialect: Dialect = Field(
        title="dialect",
        alias="dialect",
        description="Input format dialect.",
        validation_alias="dialect",
        serialization_alias="dialect",
        default=Dialect.LOCAL,
    )
    """Input format dialect."""

    version: Version = Field(
        title="version",
        alias="version",
        description="Input format version.",
        validation_alias="version",
        serialization_alias="version",
        default=Version.IV,
    )
    """Input format version."""

    seeds: (
        Annotated[int, Field(ge=1)]
        | Annotated[
            Sequence[Annotated[int, Field(ge=1, le=(1 << 32) - 1)]],
            Field(min_length=1),
        ]
    ) = Field(
        title="seeds",
        alias="seeds",
        description="Random seeds or their total number.",
        validation_alias="modelSeeds",
        serialization_alias="modelSeeds",
        default_factory=lambda: (random.getrandbits(32) or 1,),
    )
    """Random seeds or their total number."""

    entities: Sequence[Protein | RNA | DNA | Ligand] = Field(
        title="entities",
        alias="entities",
        description="Entities included in the job.",
        validation_alias="sequences",
        serialization_alias="sequences",
        default_factory=tuple,
    )
    """Entities included in the job."""

    bonds: Sequence[Bond] | None = Field(
        title="bonds",
        alias="bonds",
        description="Covalent bonds between atom pairs.",
        validation_alias="bondedAtomPairs",
        serialization_alias="bondedAtomPairs",
        default=None,
    )
    """Covalent bonds between atom pairs."""

    ccd: str | Path | None = Field(
        title="ccd",
        alias="ccd",
        description="Custom chemical components dictionary.",
        validation_alias=AliasChoices("userCCD", "userCCDPath"),
        exclude=True,
        default=None,
    )
    """Custom chemical components dictionary."""

    @computed_field(alias="userCCD", repr=False)
    @property
    def __ccd_inline(self) -> str | None:
        """Expose inline ``ccd`` for serialization.

        Returns:
            str | None: ``ccd`` when it is a string, otherwise ``None``.
        """
        return self.ccd if isinstance(self.ccd, str) else None

    @computed_field(alias="userCCDPath", repr=False)
    @property
    def __ccd_path(self) -> Path | None:
        """Expose path-based ``ccd`` for serialization.

        Returns:
            Path | None: ``ccd`` when it is a path, otherwise ``None``.
        """
        return self.ccd if isinstance(self.ccd, Path) else None

    @staticmethod
    def __allocate_ids(
        entities: Sequence[Protein | RNA | DNA | Ligand],
        ids: set[int],
    ) -> None:
        """Assign missing entity identifiers in place.

        Allocates identifiers for each entity whose :attr:`id` is unset. For
        entities with :attr:`copies` greater than one, one identifier is
        assigned per copy.

        Args:
            entities (Sequence[Protein | RNA | DNA | Ligand]): Entities that
                may require identifier assignment.
            ids (set[int]): Identifiers already reserved across the job.
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
        """Append entities to the job.

        Args:
            *entities (Protein | RNA | DNA | Ligand): One or more entities to
                add.

        Returns:
            tuple[tuple[str, ...], ...]: Identifiers of the added entities.

        Raises:
            TypeError: If no entities were provided.
        """
        if not entities:
            msg: str = "Invalid job entity: no entities were provided."
            raise TypeError(msg)

        self.entities: tuple[Protein | RNA | DNA | Ligand, ...] = (
            *tuple(self.entities),
            *entities,
        )

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
        """Load a job from an AlphaFold 3 input file.

        Args:
            path (Path): Path to the JSON input file.
            encoding (str): Text encoding used to read the file.

        Returns:
            Self: Parsed and validated job instance.
        """
        return cls.model_validate_json(
            Path(path).read_text(encoding=encoding),
        )

    def export(self: Self) -> dict[str, Any]:
        """Export the job as an AlphaFold 3 input mapping.

        Returns:
            dict[str, Any]: AlphaFold 3 input mapping.
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
        """Save the job to an AlphaFold 3 input file.

        Args:
            path (Path): Destination path for the JSON file.
            indent (int | None): JSON indentation level.
            ensure_ascii (bool): Whether to escape non-ASCII characters in the
                JSON output.
            encoding (str): Text encoding used to write the file.

        Returns:
            Path: The written path.
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
        """Coerce path-based ``ccd`` input to :class:`Path`.

        Args:
            data (Any): Raw input data.
            handler (ModelWrapValidatorHandler[Self]): Inner model validator.

        Returns:
            Self: Validated model with path-based ``ccd`` coerced to
            :class:`Path` when applicable.
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
        """Validate and assign entity identifiers.

        Ensures that explicitly provided identifiers are unique across the
        job and assigns identifiers to entities where :attr:`id` is unset.

        Args:
            entities (Sequence[Protein | RNA | DNA | Ligand]): Job entities.

        Returns:
            Sequence[Protein | RNA | DNA | Ligand]: Validated job entities.

        Raises:
            ValueError: If an identifier is used more than once.
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
        """Validate the selected input dialect.

        Args:
            value (Dialect): Requested input dialect.

        Returns:
            Dialect: Validated input dialect.

        Raises:
            NotImplementedError: If the server-side dialect is selected.
        """
        if value == Dialect.SERVER:
            msg = f"'{Dialect.SERVER}' dialect is not supported."
            raise NotImplementedError(msg)
        return value

    @model_validator(mode="after")
    def __validate_version(self: Self) -> Self:
        """Validate that ``version`` supports the configured features.

        Path-based alignments and template structures require
        :attr:`Version.II`. User-defined :attr:`ccd` paths require
        :attr:`Version.III`. Any entity :attr:`description` requires
        :attr:`Version.IV`.

        Returns:
            Self: Validated job instance.

        Raises:
            ValueError: If ``version`` is lower than the required version.
        """
        required: Version = Version.I

        if isinstance(self.ccd, Path):
            required: Version = max(required, Version.III)

        for entity in self.entities:
            if entity.description is not None:
                required: Version = max(required, Version.IV)
                continue

            if isinstance(entity, RNA) and isinstance(entity.alignment, Path):
                required: Version = max(required, Version.II)

            if isinstance(entity, Protein):
                if isinstance(entity.alignment, Path):
                    required: Version = max(required, Version.II)
                if entity.templates and any(
                    isinstance(template.structure, Path)
                    for template in entity.templates
                ):
                    required: Version = max(required, Version.II)

        if self.version < required:
            msg: str = (
                "Invalid job configuration: `version` is too low for the "
                f"selected features (version={self.version}, "
                f"required>={required})."
            )
            raise ValueError(msg)

        return self

    @field_serializer("seeds", mode="plain")
    def __expand_seeds(
        self: Self,
        value: int | Sequence[int],
    ) -> Sequence[int]:
        """Expand ``seeds`` to explicit positive 32-bit seeds.

        When :attr:`seeds` is an integer count, it is expanded at
        serialization time into a tuple of pseudo-random 32-bit seeds. When
        it is already a sequence, it is returned unchanged.

        Args:
            value (int | Sequence[int]): Random seeds or their total number.

        Returns:
            Sequence[int]: Model seeds.
        """
        if isinstance(value, int):
            value: tuple[int, ...] = tuple(
                random.getrandbits(32) or 1 for _ in range(value)
            )
        return value
