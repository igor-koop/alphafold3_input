"""DNA chain entity models.

This submodule defines :class:`DNA`, which can be used to include polymeric DNA
entities in an AlphaFold 3 input.

Exports:
    - :class:`DNA`: DNA chain entity model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Self

from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StringConstraints,
    field_validator,
    model_serializer,
    model_validator,
)

from .modification import Entity, Modification

__all__: list[str] = [
    "DNA",
]


class DNA(BaseModel):
    """DNA chain entity specification.

    A DNA chain is defined by its nucleotide :attr:`sequence`, optional
    :attr:`modifications`, and one or more chain identifiers via :attr:`id`.
    Multiple copies can be defined either by setting :attr:`copies` or by
    providing multiple explicit identifiers in :attr:`id`.

    The optional :attr:`description` field is supported only when
    :attr:`Job.version` is set to :attr:`Version.IV`.

    Attributes:
        id (str | Sequence[str] | None): DNA chain identifier(s).
        description (str | None): Free-text DNA chain description.
        sequence (str): DNA chain nucleotide sequence.
        modifications (Sequence[Modification]): DNA chain residue
            modifications.
        copies (int): Number of DNA chain copies.

    Examples:
        DNA chain with a description.

        >>> DNA(
        ...     description="Promoter for bacteriophage T7 RNA polymerase",
        ...     sequence="TAATACGACTCACTATAGG",
        ... )

        Multiple copies of a DNA chain.

        >>> DNA(
        ...     sequence="GAATTC",
        ...     copies=2,
        ... )

        DNA chain with modified residues.

        >>> heptamer = DNA(sequence="GACCTCT")
        >>> heptamer.modify(
        ...     Modification(type="6OG", position=1),
        ...     Modification(type="6MA", position=2),
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
        description="DNA chain identifier(s).",
        min_length=1,
        validation_alias=AliasPath("dna", "id"),
        serialization_alias="id",
        default=None,
    )
    """DNA chain identifier(s)."""

    description: str | None = Field(
        title="description",
        alias="description",
        description="Free-text DNA chain description.",
        validation_alias=AliasPath("dna", "description"),
        serialization_alias="description",
        default=None,
    )
    """Free-text DNA chain description."""

    sequence: Annotated[str, StringConstraints(pattern="^[ACGT]+$")] = Field(
        title="sequence",
        alias="sequence",
        description="DNA chain nucleotide sequence.",
        min_length=1,
        validation_alias=AliasPath("dna", "sequence"),
        serialization_alias="sequence",
    )
    """DNA chain nucleotide sequence."""

    modifications: Sequence[Modification] = Field(
        title="modifications",
        alias="modifications",
        description="DNA chain residue modifications.",
        validation_alias=AliasPath("dna", "modifications"),
        serialization_alias="modifications",
        default_factory=tuple,
    )
    """DNA chain residue modifications."""

    copies: int = Field(
        title="copies",
        alias="copies",
        description="Number of DNA chain copies.",
        ge=1,
        validation_alias="copies",
        exclude=True,
        repr=False,
        default=1,
    )
    """Number of DNA chain copies."""

    def modify(self: Self, *modifications: Modification) -> Self:
        """Append residue modifications to the DNA chain.

        Args:
            *modifications (Modification): One or more modifications to add.

        Returns:
            Self: DNA chain with appended modifications.

        Raises:
            TypeError: If no modifications were provided.
        """
        if not modifications:
            msg: str = (
                "Invalid DNA modification: no modifications were provided."
            )
            raise TypeError(msg)

        self.modifications: tuple[Modification, ...] = tuple(
            self.modifications,
        ) + tuple(modifications)
        return self

    @field_validator("modifications", mode="after")
    @classmethod
    def __scope_modifications(
        cls: type[Self],
        value: Sequence[Modification],
    ) -> Sequence[Modification]:
        """Assign DNA scope to each modification.

        Args:
            value (Sequence[Modification]): Modification entries.

        Returns:
            Sequence[Modification]: Scoped modification entries.
        """
        for modification in value:
            object.__setattr__(modification, "scope", Entity.DNA)
        return value

    @model_validator(mode="after")
    def __validate_modifications(self: Self) -> Self:
        """Validate residue modifications against the DNA sequence.

        Ensures that each modification position is within the sequence and that
        no residue is modified more than once.

        Returns:
            Self: Validated DNA chain instance.

        Raises:
            ValueError: If a modification position is out of range.
            ValueError: If multiple modifications target the same position.
        """
        length: int = len(self.sequence)
        modified: set[int] = set()

        for modification in self.modifications:
            if modification.position > length:
                msg: str = (
                    "Invalid DNA modification: modification `position` is out "
                    f"of range (position={modification.position}, "
                    f"len(sequence)={length})."
                )
                raise ValueError(msg)

            if modification.position in modified:
                msg: str = (
                    "Invalid DNA modification list: multiple modifications on "
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
            Self: Validated DNA chain instance.

        Raises:
            ValueError: If ``copies`` is inconsistent with ``id``.
        """
        if self.id is None:
            return self

        n: int = len(self.id) if not isinstance(self.id, str) else 1

        if "copies" in self.model_fields_set and self.copies != n:
            msg: str = (
                "Conflicting DNA configuration: `copies` is inconsistent "
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
