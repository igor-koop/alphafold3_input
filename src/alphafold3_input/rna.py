"""RNA chain entity models.

This submodule defines :class:`RNA`, which can be used to include
polymeric RNA entities in an AlphaFold 3 input.

Exports:
    - :class:`RNA`: RNA chain entity model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    AliasChoices,
    AliasPath,
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

__all__: list[str] = [
    "RNA",
]


class RNA(BaseModel):
    """RNA chain entity specification.

    An RNA chain is defined by its nucleotide :attr:`sequence`, optional
    residue :attr:`modifications`, and optional multiple sequence
    :attr:`alignment`.

    Modified residues can be provided through :class:`Modification` entries.
    Additional entries may be appended using :meth:`modify`.

    A multiple sequence alignment may be provided either inline as a string or
    as a filesystem path.

    Multiple copies of an RNA chain can be defined either by setting
    :attr:`copies` or by providing multiple explicit identifiers in
    :attr:`id`. The optional :attr:`description` field is supported only when
    :attr:`Job.version` is set to :attr:`Version.IV`.

    Attributes:
        id (str | Sequence[str] | None): RNA chain identifier(s).
        description (str | None): Free-text RNA chain description.
        sequence (str): RNA chain nucleotide sequence.
        modifications (Sequence[Modification]): RNA chain residue
            modifications.
        alignment (str | Path | None): Multiple sequence alignment.
        copies (int): Number of RNA chain copies.

    Examples:
        RNA chain with a description.

        >>> RNA(
        ...     description="Ribosome-binding site from T7 phage, gene 10",
        ...     sequence="UUAACUUUAAGAAGGAG",
        ... )

        Multiple copies of an RNA chain.

        >>> RNA(
        ...     sequence="AAGGACGGGUCC",
        ...     copies=2,
        ... )

        RNA chain with modified residues.

        >>> tetramer = RNA(sequence="AGCU")
        >>> tetramer.modify(
        ...     Modification(type="2MG", position=1),
        ...     Modification(type="5MC", position=4),
        ... )

        RNA chain with an alignment file.

        >>> RNA(
        ...     id=["A", "B"],
        ...     sequence="ACAUGAGGAUCACCCAUGU",
        ...     alignment=Path("alignment.a3m"),
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

    id: (
        Annotated[str, StringConstraints(pattern="^[A-Z]+$")]
        | Sequence[Annotated[str, StringConstraints(pattern="^[A-Z]+$")]]
        | None
    ) = Field(
        title="id",
        alias="id",
        description="RNA chain identifier(s).",
        min_length=1,
        validation_alias=AliasPath("rna", "id"),
        serialization_alias="id",
        default=None,
    )
    """RNA chain identifier(s)."""

    description: str | None = Field(
        title="description",
        alias="description",
        description="Free-text RNA chain description.",
        validation_alias=AliasPath("rna", "description"),
        serialization_alias="description",
        default=None,
    )
    """Free-text RNA chain description."""

    sequence: Annotated[str, StringConstraints(pattern="^[ACGU]+$")] = Field(
        title="sequence",
        alias="sequence",
        description="RNA chain nucleotide sequence.",
        min_length=1,
        validation_alias=AliasPath("rna", "sequence"),
        serialization_alias="sequence",
    )
    """RNA chain nucleotide sequence."""

    modifications: Sequence[Modification] = Field(
        title="modifications",
        alias="modifications",
        description="RNA chain residue modifications.",
        validation_alias=AliasPath("rna", "modifications"),
        serialization_alias="modifications",
        default_factory=tuple,
    )
    """RNA chain residue modifications."""

    alignment: str | Path | None = Field(
        title="alignment",
        alias="alignment",
        description="Multiple sequence alignment for RNA chain.",
        validation_alias=AliasChoices(
            AliasPath("rna", "unpairedMsa"),
            AliasPath("rna", "unpairedMsaPath"),
        ),
        exclude=True,
        default=None,
    )
    """Multiple sequence alignment for RNA chain."""

    copies: int = Field(
        title="copies",
        alias="copies",
        description="Number of RNA chain copies.",
        ge=1,
        validation_alias="copies",
        exclude=True,
        repr=False,
        default=1,
    )
    """Number of RNA chain copies."""

    @computed_field(alias="unpairedMsa", repr=False)
    @property
    def __alignment_inline(self) -> str | None:
        """Expose inline ``alignment`` for serialization.

        Returns:
            str | None: ``alignment`` when it is a string, otherwise ``None``.
        """
        return self.alignment if isinstance(self.alignment, str) else None

    @computed_field(alias="unpairedMsaPath", repr=False)
    @property
    def __alignment_path(self) -> Path | None:
        """Expose path-based ``alignment`` for serialization.

        Returns:
            Path | None: ``alignment`` when it is a path, otherwise ``None``.
        """
        return self.alignment if isinstance(self.alignment, Path) else None

    def modify(self: Self, *modifications: Modification) -> Self:
        """Append residue modifications to the RNA chain.

        Args:
            *modifications (Modification): One or more modifications to add.

        Returns:
            Self: RNA chain with appended modifications.

        Raises:
            TypeError: If no modifications were provided.
        """
        if not modifications:
            msg: str = (
                "Invalid RNA modification: no modifications were provided."
            )
            raise TypeError(msg)

        self.modifications: tuple[Modification, ...] = tuple(
            self.modifications,
        ) + tuple(modifications)
        return self

    @model_validator(mode="wrap")
    @classmethod
    def __coerce_alignment(
        cls: type[Self],
        data: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Coerce path-based ``alignment`` input to :class:`Path`.

        Args:
            data (Any): Raw input data.
            handler (ModelWrapValidatorHandler[Self]): Inner model validator.

        Returns:
            Self: Validated model with path-based ``alignment`` coerced to
            :class:`Path` when applicable.
        """
        if not isinstance(data, Mapping):
            return handler(data)

        alias: str = "unpairedMsaPath"
        value: str | None = None

        if isinstance(data.get("rna"), Mapping):
            value: str | None = data["rna"].get(alias)
        else:
            value: str | None = data.get(alias)

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
        """Assign RNA scope to each modification.

        Args:
            value (Sequence[Modification]): Modification entries.

        Returns:
            Sequence[Modification]: Scoped modification entries.
        """
        for modification in value:
            object.__setattr__(modification, "scope", Entity.RNA)
        return value

    @model_validator(mode="after")
    def __validate_modifications(self: Self) -> Self:
        """Validate residue modifications against the RNA sequence.

        Ensures that each modification position is within the sequence and that
        no residue is modified more than once.

        Returns:
            Self: Validated RNA chain instance.

        Raises:
            ValueError: If a modification position is out of range.
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
        """Validate consistency between ``id`` and ``copies``.

        If :attr:`id` is provided, :attr:`copies` is set to ``len(id)``. If
        :attr:`copies` was explicitly provided and is inconsistent with
        :attr:`id`, a :class:`ValueError` is raised.

        Returns:
            Self: Validated RNA chain instance.

        Raises:
            ValueError: If ``copies`` is inconsistent with ``id``.
        """
        if self.id is None:
            return self

        n: int = len(self.id) if not isinstance(self.id, str) else 1

        if "copies" in self.model_fields_set and self.copies != n:
            msg: str = (
                "Conflicting RNA configuration: `copies` is inconsistent "
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
        """Serialize the entity in wrapped AlphaFold 3 form.

        Returns:
            dict[str, Any]: Wrapped entity mapping.
        """
        return {self.__class__.__name__.lower(): handler(self)}
