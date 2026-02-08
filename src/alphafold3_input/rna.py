"""RNA chain entity model.

This submodule defines the `RNA` model, which can be used to include
polymeric RNA entities in an AlphaFold 3 input for structure prediction.

Exports:
    RNA: Model representing an RNA chain entity.
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
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

from .modification import Entity, Modification

__all__: list[str] = [
    "RNA",
]


class RNA(BaseModel):
    """RNA chain entity specification.

    Represents a polymeric RNA entity for inclusion under `Job.sequences` in an
    AlphaFold 3 input.

    An RNA chain is defined by its nucleotide `sequence` (A, C, G, and U).
    Modified residues can be provided through `modifications` as `Modification`
    entries. Additional entries may be appended using `RNA.modify()`.

    An RNA chain may include a multiple sequence alignment via `alignment`,
    either provided inline as a string or as a filesystem path.

    Multiple copies of an RNA chain can be defined either by setting `copies`
    or by providing multiple explicit identifiers via `id`. Optional
    `description` is supported only when `Job.version` is set to input format
    version 4.

    Attributes:
        id (Sequence[str] | None): RNA chain identifier(s).
        description (str | None): Free-text RNA chain description.
        sequence (str): RNA chain nucleotide sequence.
        modifications (Sequence[Modification]): RNA chain residue
            modifications.
        alignment (str | Path | None): Multiple sequence alignment for RNA
            chain.
        copies (int): Number of RNA chain copies.

    Examples:
        RNA chain with a description.
        ```python
        RNA(
            description="Ribosome-binding site from T7 phage, gene 10",
            sequence="UUAACUUUAAGAAGGAG",
        )
        ```

        Multiple copies of an RNA chain.
        ```python
        RNA(
            sequence="AAGGACGGGUCC",
            copies=2
        )
        ```

        RNA chain with modified residues.
        ```python
        tetramer = RNA(sequence="AGCU")
        tetramer.modify(
            Modification(type="2MG", position=1),
            Modification(type="5MC", position=4),
        )
        ```

        RNA chain with an alignment file.
        ```python
        RNA(
            id=["A", "B"],
            sequence="ACAUGAGGAUCACCCAUGU",
            alignment=Path("alignment.a3m"),
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
            description="RNA chain identifier(s).",
            min_length=1,
            validation_alias="id",
            serialization_alias="id",
        ),
    ] = None

    description: Annotated[
        str | None,
        Field(
            title="description",
            description="Free-text RNA chain description.",
            validation_alias="description",
            serialization_alias="description",
        ),
    ] = None

    sequence: Annotated[
        Annotated[str, StringConstraints(pattern="^[ACGU]+$")],
        Field(
            title="sequence",
            description="RNA chain nucleotide sequence.",
            min_length=1,
            validation_alias="sequence",
            serialization_alias="sequence",
        ),
    ]

    modifications: Annotated[
        Sequence[Modification],
        Field(
            title="modifications",
            description="RNA chain residue modifications.",
            validation_alias="modifications",
            serialization_alias="modifications",
        ),
    ] = Field(default_factory=tuple)

    alignment: Annotated[
        str | Path | None,
        Field(
            title="alignment",
            description="Multiple sequence alignment for RNA chain.",
            validation_alias=AliasChoices("unpairedMsa", "unpairedMsaPath"),
            exclude=True,
        ),
    ] = None

    copies: Annotated[
        int,
        Field(
            title="copies",
            description="Number of RNA chain copies.",
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

    def modify(self: Self, *modifications: Modification) -> Self:
        """Append one or more residue modifications to an RNA chain.

        Args:
            *modifications (Modification): One or more modifications to add.

        Returns:
            out (Self): RNA chain entity with modified residues.

        Raises:
            TypeError: If no modifications were provided.

        """
        if not modifications:
            msg: str = (
                "Invalid RNA modification: no modifications were provided."
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
        """Assign RNA scope to `modifications`.

        `Modification` model supports multiple polymer contexts, so each entry
        in `modifications` is tagged with `scope=Entity.RNA` to ensure correct
        serialization.

        Args:
            value (Sequence[Modification]): Modification entries.

        Returns:
            out (Sequence[Modification]): Scoped modification entries.

        """
        for modification in value:
            object.__setattr__(modification, "scope", Entity.RNA)
        return value

    @model_validator(mode="after")
    def __validate_modifications(self: Self) -> Self:
        """Validate residue `modifications` against the RNA chain `sequence`.

        Ensures that each modification targets a valid residue position within
        the RNA chain `sequence`, and that no residue is modified more than
        once.

        Returns:
            out (Self): The validated RNA chain instance.

        Raises:
            ValueError: If a modification targets a position outside the
                RNA chain `sequence`.
            ValueError: If multiple modifications target the same position.

        """
        length: int = len(self.sequence)
        modified: set[int] = set()

        for modification in self.modifications:
            if modification.position > length:
                msg: str = (
                    "Invalid RNA modification: modification `position` is out "
                    f"of range (position={modification.position}, "
                    f"len(sequence)={length})."
                )
                raise ValueError(msg)

            if modification.position in modified:
                msg: str = (
                    "Invalid RNA modification list: multiple modifications on "
                    f"the same `position` (position={modification.position})."
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
            out (Self): The validated RNA chain instance.

        Raises:
            ValueError: If `copies` was explicitly provided and is inconsistent
                with `id`.

        """
        if self.id is None:
            return self

        n: int = len(self.id)

        if "copies" in self.model_fields_set and self.copies != n:
            msg: str = (
                "Conflicting RNA configuration: `copies` is inconsistent "
                f"with the length of `id` (copies={self.copies}, len(id)={n})."
            )
            raise ValueError(msg)

        object.__setattr__(self, "copies", n)
        return self
