"""Residue modification models.

This submodule defines :class:`Modification`, which represents a single
residue modification on polymeric entities in an AlphaFold 3 input. It also
provides :class:`Entity` for selecting the modification scope.

Exports:
    - :class:`Modification`: Residue modification model.
    - :class:`Entity`: AlphaFold 3 entity type enum.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)

__all__: list[str] = [
    "Entity",
    "Modification",
]


class Entity(StrEnum):
    """AlphaFold 3 entity type identifier."""

    PROTEIN = "protein"
    """Protein polymer entity."""

    RNA = "rna"
    """RNA polymer entity."""

    DNA = "dna"
    """DNA polymer entity."""

    LIGAND = "ligand"
    """Ligand non-polymer entity."""


class Modification(BaseModel):
    """Residue modification specification.

    A modification is defined by its CCD :attr:`type` and a 1-based
    :attr:`position` within the parent entity sequence.

    The :attr:`scope` selects the polymer context used for serialization.
    It is typically assigned automatically by the parent entity model, such
    as :class:`DNA`, :class:`RNA`, or :class:`Protein`.

    Attributes:
        scope (Entity | None): Modification polymer context.
        type (str): Modification CCD code.
        position (int): Modification position in the parent sequence.

    Examples:
        Base modification.

        >>> methylation = Modification(type="5MC", position=4)

        Post-translational modification.

        >>> ptm = Modification(type="HY3", position=1)
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
        use_enum_values=False,
    )

    scope: Literal[Entity.PROTEIN, Entity.DNA, Entity.RNA] | None = Field(
        title="scope",
        alias="scope",
        description="Modification polymer context.",
        validation_alias="scope",
        exclude=True,
        repr=False,
        default=None,
    )
    """Modification polymer context."""

    type: str = Field(
        title="type",
        alias="type",
        description="Modification CCD code.",
        validation_alias=AliasChoices("ptmType", "modificationType"),
        exclude=True,
    )
    """Modification CCD code."""

    position: int = Field(
        title="position",
        alias="position",
        description="Modification position in the parent sequence.",
        ge=1,
        validation_alias=AliasChoices("ptmPosition", "basePosition"),
        exclude=True,
    )
    """Modification position in the parent sequence."""

    @computed_field(alias="ptmType", repr=False)
    @property
    def __type_ptm(self) -> str | None:
        """Expose protein modification ``type`` for serialization.

        Returns:
            str | None: ``type`` when :attr:`scope` is :attr:`Entity.PROTEIN`,
            otherwise ``None``.
        """
        return self.type if self.scope == Entity.PROTEIN else None

    @computed_field(alias="ptmPosition", repr=False)
    @property
    def __position_ptm(self) -> int | None:
        """Expose protein modification ``position`` for serialization.

        Returns:
            int | None: ``position`` when :attr:`scope` is
            :attr:`Entity.PROTEIN`, otherwise ``None``.
        """
        return self.position if self.scope == Entity.PROTEIN else None

    @computed_field(alias="modificationType", repr=False)
    @property
    def __type_base(self) -> str | None:
        """Expose nucleic-acid modification ``type`` for serialization.

        Returns:
            str | None: ``type`` when :attr:`scope` is :attr:`Entity.DNA` or
            :attr:`Entity.RNA`, otherwise ``None``.
        """
        return self.type if self.scope in (Entity.RNA, Entity.DNA) else None

    @computed_field(alias="basePosition", repr=False)
    @property
    def __position_base(self) -> int | None:
        """Expose nucleic-acid modification ``position`` for serialization.

        Returns:
            int | None: ``position`` when :attr:`scope` is :attr:`Entity.DNA`
            or :attr:`Entity.RNA`, otherwise ``None``.
        """
        return self.position if self.scope in (Entity.RNA, Entity.DNA) else None

    @model_serializer(mode="wrap")
    def __scoped_serialization(
        self: Self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Serialize the modification using its assigned scope.

        Returns:
            dict[str, Any]: Scoped serialized modification mapping.

        Raises:
            ValueError: If :attr:`scope` is unset at serialization time.
        """
        if self.scope is None:
            msg: str = (
                "Invalid modification: `scope` must be set for serialization."
            )
            raise ValueError(msg)
        return handler(self)
