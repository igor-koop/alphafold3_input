"""DNA chain entity model.

This submodule defines the `DNA` model, which can be used to include
polymeric DNA entities in an AlphaFold 3 input for structure prediction.

Exports:
    DNA: Model representing a DNA chain entity.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
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

    Represents a polymeric DNA entity for inclusion under `Job.sequences` in an
    AlphaFold 3 input.

    A DNA chain is defined by its nucleotide `sequence` (A, C, G, and T).
    Modified residues can be provided through `modifications` as `Modification`
    entries. Additional entries may be appended using `DNA.modify()`.

    Multiple copies of a DNA chain can be defined either by setting `copies` or
    by providing multiple explicit identifiers via `id`. Optional `description`
    is supported only when `Job.version` is set to input format version 4.

    Attributes:
        id (Sequence[str] | None): DNA chain identifier(s).
        description (str | None): Free-text DNA chain description.
        sequence (str): DNA chain nucleotide sequence.
        modifications (Sequence[Modification]): DNA chain residue
            modifications.
        copies (int): Number of DNA chain copies.

    Examples:
        DNA chain with a description.
        ```python
        DNA(
            description="Promoter for bacteriophage T7 RNA polymerase",
            sequence="TAATACGACTCACTATAGG",
        )
        ```

        Multiple copies of a DNA chain.
        ```python
        DNA(
            sequence="GAATTC",
            copies=2
        )
        ```

        DNA chain with modified residues.
        ```python
        heptamer = DNA(sequence="GACCTCT")
        heptamer.modify(
            Modification(type="6OG", position=1),
            Modification(type="6MA", position=2),
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
            description="DNA chain identifier(s).",
            min_length=1,
            validation_alias=AliasPath("dna", "id"),
            serialization_alias="id",
        ),
    ] = None

    description: Annotated[
        str | None,
        Field(
            title="description",
            description="Free-text DNA chain description.",
            validation_alias=AliasPath("dna", "description"),
            serialization_alias="description",
        ),
    ] = None

    sequence: Annotated[
        Annotated[str, StringConstraints(pattern="^[ACGT]+$")],
        Field(
            title="sequence",
            description="DNA chain nucleotide sequence.",
            min_length=1,
            validation_alias=AliasPath("dna", "sequence"),
            serialization_alias="sequence",
        ),
    ]

    modifications: Annotated[
        Sequence[Modification],
        Field(
            title="modifications",
            description="DNA chain residue modifications.",
            validation_alias=AliasPath("dna", "modifications"),
            serialization_alias="modifications",
        ),
    ] = Field(default_factory=tuple)

    copies: Annotated[
        int,
        Field(
            title="copies",
            description="Number of DNA chain copies.",
            ge=1,
            validation_alias="copies",
            exclude=True,
            repr=False,
        ),
    ] = 1

    def modify(self: Self, *modifications: Modification) -> Self:
        """Append one or more residue modifications to a DNA chain.

        Args:
            *modifications (Modification): One or more modifications to add.

        Returns:
            out (Self): DNA chain entity with modified residues.

        Raises:
            TypeError: If no modifications were provided.

        """
        if not modifications:
            msg: str = (
                "Invalid DNA modification: no modifications were provided."
            )
            raise TypeError(msg)

        self.modifications = tuple(self.modifications) + tuple(modifications)
        return self

    @field_validator("modifications", mode="after")
    @classmethod
    def __scope_modifications(
        cls: type[Self],
        value: Sequence[Modification],
    ) -> Sequence[Modification]:
        """Assign DNA scope to `modifications`.

        `Modification` model supports multiple polymer contexts, so each entry
        in `modifications` is tagged with `scope=Entity.DNA` to ensure correct
        serialization.

        Args:
            value (Sequence[Modification]): Modification entries.

        Returns:
            out (Sequence[Modification]): Scoped modification entries.

        """
        for modification in value:
            object.__setattr__(modification, "scope", Entity.DNA)
        return value

    @model_validator(mode="after")
    def __validate_modifications(self: Self) -> Self:
        """Validate residue `modifications` against the DNA chain `sequence`.

        Ensures that each modification targets a valid residue position within
        the DNA chain `sequence`, and that no residue is modified more than
        once.

        Returns:
            out (Self): The validated DNA chain instance.

        Raises:
            ValueError: If a modification targets a position outside the
                DNA chain `sequence`.
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
        """Enforce consistency between `id` and `copies`.

        If `id` is provided, `copies` is set to `len(id)`. If `copies` was
        explicitly provided and is inconsistent with `id`, a `ValueError` is
        raised.

        Returns:
            out (Self): The validated DNA chain instance.

        Raises:
            ValueError: If `copies` was explicitly provided and is inconsistent
                with `id`.

        """
        if self.id is None:
            return self

        n: int = len(self.id)

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
        """Serialize the entity using the AlphaFold 3 wrapped representation.

        Returns:
            out (dict[str, Any]): Wrapped entity mapping suitable for the
                AlphaFold 3 input format.

        """
        return {self.__class__.__name__.lower(): handler(self)}
