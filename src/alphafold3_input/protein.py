"""Protein chain entity model.

This submodule defines the `Protein` model, which can be used to include
polymeric protein entities in an AlphaFold 3 input for structure prediction.

Exports:
    Protein: Model representing a protein chain entity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    SerializerFunctionWrapHandler,
    StringConstraints,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

from .modification import Entity, Modification
from .template import Template

__all__: list[str] = [
    "Protein",
]


class Protein(BaseModel):
    """Protein chain entity specification.

    Represents a polymeric protein entity for inclusion under `Job.sequences`
    in an AlphaFold 3 input.

    A protein chain is defined by its amino acid `sequence` (20 standard
    proteinogenic amino acids; 1-letter code). Post-translational modifications
    can be provided through `modifications` as `Modification` entries.
    Additional entries may be appended using `Protein.modify()`.

    A protein chain may include a multiple sequence alignment via `alignment`,
    either provided inline as a string or as a filesystem path. MSA pairing,
    relevant for multimer folding, is set automatically: when `alignment` is
    provided, it is serialized as an empty string and is otherwise omitted.

    Structural templates can be provided via `templates` as `Template` entries.

    Multiple copies of a protein chain can be defined either by setting
    `copies` or by providing multiple explicit identifiers via `id`. Optional
    `description` is supported only when `Job.version` is set to input format
    version 4.

    Attributes:
        id (Sequence[str] | None): Protein chain identifier(s).
        description (str | None): Free-text protein chain description.
        sequence (str): Protein chain amino acid sequence.
        modifications (Sequence[Modification]): Protein chain
            post-translational modifications.
        alignment (str | Path | None): Multiple sequence alignment for protein
            chain.
        templates (Sequence[Template]): Structural templates for protein chain.
        copies (int): Number of protein chain copies.

    Examples:
        Protein chain with a description.
        ```python
        Protein(
            description="AviTag for BirA-mediated biotinylation",
            sequence="GLNDIFEAQKIEWHE",
        )
        ```

        Multiple copies of a protein chain.
        ```python
        Protein(
            sequence="RMKQLEDKVEELLSKKYHLENEVARLKKLVGER",
            copies=2
        )
        ```

        Protein chain with modified residues.
        ```python
        peptide = Protein(sequence="PVLSCGEWQL")
        peptide.modify(
            Modification(type="HY3", position=1),
            Modification(type="P1L", position=5),
        )
        ```

        Protein chain with an alignment file.
        ```python
        Protein(
            id=["A", "B"],
            sequence="KRRWKKNFIAVSAANRFKKISSSGAL",
            alignment=Path("alignment.a3m"),
        )
        ```

        Protein chain with a structural template.
        ```python
        Protein(
            id=["C"],
            sequence="RPACQLW",
            templates=[
                Template(
                    structure=Path("template.cif.gz"),
                    indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
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

    id: Annotated[
        Sequence[Annotated[str, StringConstraints(pattern="^[A-Z]+$")]] | None,
        Field(
            title="id",
            description="Protein chain identifier(s).",
            min_length=1,
            validation_alias="id",
            serialization_alias="id",
        ),
    ] = None

    description: Annotated[
        str | None,
        Field(
            title="description",
            description="Free-text protein chain description.",
            validation_alias="description",
            serialization_alias="description",
        ),
    ] = None

    sequence: Annotated[
        Annotated[str, StringConstraints(pattern="^[ACDEFGHIKLMNPQRSTVWY]+$")],
        Field(
            title="sequence",
            description="Protein chain amino acid sequence.",
            min_length=1,
            validation_alias="sequence",
            serialization_alias="sequence",
        ),
    ]

    modifications: Annotated[
        Sequence[Modification],
        Field(
            title="modifications",
            description="Protein chain residue modifications.",
            validation_alias="modifications",
            serialization_alias="modifications",
        ),
    ] = Field(default_factory=tuple)

    alignment: Annotated[
        str | Path | None,
        Field(
            title="alignment",
            description="Multiple sequence alignment for protein chain.",
            validation_alias=AliasChoices("unpairedMsa", "unpairedMsaPath"),
            exclude=True,
        ),
    ] = None

    templates: Annotated[
        Sequence[Template],
        Field(
            title="templates",
            description="Structural templates for protein chain.",
            validation_alias="templates",
            serialization_alias="templates",
        ),
    ] = Field(default_factory=tuple)

    copies: Annotated[
        int,
        Field(
            title="copies",
            description="Number of protein chain copies.",
            ge=1,
            validation_alias="copies",
            exclude=True,
            repr=False,
        ),
    ] = 1

    @computed_field(alias="unpairedMsa", repr=False)
    @property
    def __alignment_inline(self) -> str | None:
        """Expose inline `alignment` for serialization.

        Returns:
            out (str | None): `alignment` when it is a string,
                otherwise `None`.

        """
        return self.alignment if isinstance(self.alignment, str) else None

    @computed_field(alias="unpairedMsaPath", repr=False)
    @property
    def __alignment_path(self) -> Path | None:
        """Expose `alignment` path for serialization.

        Returns:
            out (Path | None): `alignment` when it is a path, otherwise
                `None`.

        """
        return self.alignment if isinstance(self.alignment, Path) else None

    @computed_field(alias="pairedMsa", repr=False)
    @property
    def __alignment_paired(self) -> str | None:
        """Sets MSA pairing based on `alignment` for serialization.

        Returns:
            out (str | None): Empty string when `alignment` is provided,
                otherwise `None`.

        """
        return "" if self.alignment is not None else None

    def modify(self: Self, *modifications: Modification) -> Self:
        """Append one or more residue modifications to a protein chain.

        Args:
            *modifications (Modification): One or more modifications to add.

        Returns:
            out (Self): Protein chain entity with modified residues.

        Raises:
            TypeError: If no modifications were provided.

        """
        if not modifications:
            msg: str = (
                "Invalid protein modification: no modifications were provided."
            )
            raise TypeError(msg)

        self.modifications = tuple(self.modifications) + tuple(modifications)
        return self

    @model_validator(mode="wrap")
    @classmethod
    def __coerce_alignment(
        cls: type[Self],
        data: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Coerce `alignment` to a `Path`.

        Inspects the raw input data. When the input is a mapping containing
        `unpairedMsaPath`, its value is converted to a `Path` and assigned to
        `alignment` after successful model validation.

        Args:
            data (Any): Raw input data.
            handler (ModelWrapValidatorHandler[Self]): Inner model validator.

        Returns:
            out (Self): Validated model with `alignment` coerced to `Path` when
                applicable.

        """
        if not isinstance(data, Mapping):
            return handler(data)

        alias: str = "unpairedMsaPath"
        value: object | None = data.get(alias)
        model: Self = handler(data)
        if value is not None and isinstance(value, str):
            object.__setattr__(model, "alignment", Path(value))
        return model

    @field_validator("modifications", mode="after")
    @classmethod
    def __scope_modifications(
        cls: type[Self],
        value: Sequence[Modification],
    ) -> Sequence[Modification]:
        """Assign protein scope to `modifications`.

        `Modification` model supports multiple polymer contexts, so each entry
        in `modifications` is tagged with `scope=Entity.PROTEIN` to ensure
        correct serialization.

        Args:
            value (Sequence[Modification]): Modification entries.

        Returns:
            out (Sequence[Modification]): Scoped modification entries.

        """
        for modification in value:
            object.__setattr__(modification, "scope", Entity.PROTEIN)
        return value

    @model_validator(mode="after")
    def __validate_modifications(self: Self) -> Self:
        """Validate post-translational `modifications` against the protein chain `sequence`.

        Ensures that each modification targets a valid residue position within
        the protein chain `sequence`, and that no residue is modified more than
        once.

        Returns:
            out (Self): The validated protein chain instance.

        Raises:
            ValueError: If a modification targets a position outside the
                protein chain `sequence`.
            ValueError: If multiple modifications target the same position.

        """  # noqa: E501
        length: int = len(self.sequence)
        modified: set[int] = set()

        for modification in self.modifications:
            if modification.position > length:
                msg: str = (
                    "Invalid protein modification: modification `position` is "
                    f"out of range (position={modification.position}, "
                    f"len(sequence)={length})."
                )
                raise ValueError(msg)

            if modification.position in modified:
                msg: str = (
                    "Invalid protein modification list: multiple "
                    "modifications on the same `position` "
                    f"(position={modification.position})."
                )
                raise ValueError(msg)

            modified.add(modification.position)

        return self

    @model_validator(mode="after")
    def __validate_copies(self: Self) -> Self:
        """Enforce consistency between `id` and `copies`.

        If `id` is provided, `copies` is set to `len(id)`. If `copies` was
        explicitly provided and is inconsistent with `id`, a `ValueError` is
        raised.

        Returns:
            out (Self): The validated protein chain instance.

        Raises:
            ValueError: If `copies` was explicitly provided and is inconsistent
                with `id`.

        """
        if self.id is None:
            return self

        n: int = len(self.id)

        if "copies" in self.model_fields_set and self.copies != n:
            msg: str = (
                "Conflicting protein configuration: `copies` is inconsistent "
                f"with the length of `id` (copies={self.copies}, len(id)={n})."
            )
            raise ValueError(msg)

        object.__setattr__(self, "copies", n)
        return self

    @model_serializer(mode="wrap")
    def __wrapped_serialization(
        self: Self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Serialize the entity using the AlphaFold 3 wrapped representation.

        Returns:
            out (dict[str, Any]): Wrapped entity mapping suitable for the
                AlphaFold 3 input format.

        """
        return {self.__class__.__name__.lower(): handler(self)}
