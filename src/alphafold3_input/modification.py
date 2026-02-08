"""Residue modification model.

This submodule defines the `Modification` model, which represents a single
residue modification on polymeric entities (protein, DNA, or RNA) in an
AlphaFold 3 input for structure prediction. The module also provides `Entity`
enum to control modification scope.

Exports:
    Modification: Model representing a residue modification.
    Entity: Enum selecting the AlphaFold 3 entity types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

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
    """AlphaFold 3 entity type identifier.

    Members:
        PROTEIN: Protein polymer entity (`"protein"`).
        RNA: RNA polymer entity (`"rna"`).
        DNA: DNA polymer entity (`"dna"`).
        LIGAND: Ligand non-polymer entity (`"ligand"`).

    """

    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    LIGAND = "ligand"


class Modification(BaseModel):
    """Residue modification specification.

    Represents a single residue modification targeting a polymeric entity.
    The modification `type` is defined by its CCD code and a 1-based residue
    `position` referring to the parent entity's `sequence`.

    `Modification` supports multiple polymer contexts via `scope`. The `scope`
    is typically assigned by the parent entity model (e.g., `DNA`, `RNA`,
    `Protein`) to ensure the correct AlphaFold 3 field names are produced
    during serialization.

    Attributes:
        scope (Entity | None): Modification polymer context.
        type (str): Modification CCD code.
        position (int): Modification position in the parent sequence.

    Examples:
        Base modification.
        ```python
        methylation = Modification(type="5MC", position=4)
        ```

        Post-translational modification.
        ```python
        ptm = Modification(type="HY3", position=1)
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

    scope: Annotated[
        Literal[Entity.PROTEIN, Entity.DNA, Entity.RNA] | None,
        Field(
            title="scope",
            description="Modification polymer context.",
            validation_alias="scope",
            exclude=True,
            repr=False,
        ),
    ] = None

    type: Annotated[
        str,
        Field(
            title="type",
            description="Modification CCD code.",
            validation_alias=AliasChoices("ptmType", "modificationType"),
            exclude=True,
        ),
    ]

    position: Annotated[
        int,
        Field(
            title="position",
            description="Modification position in the parent sequence.",
            ge=1,
            validation_alias=AliasChoices("ptmPosition", "basePosition"),
            exclude=True,
        ),
    ]

    @computed_field(alias="ptmType", repr=False)
    @property
    def __type_ptm(self) -> str | None:
        """Expose protein modification `type` for serialization.

        Returns:
            out (str | None): `type` when `scope` is `Entity.PROTEIN`,
                otherwise `None`.

        """
        return self.type if self.scope == Entity.PROTEIN else None

    @computed_field(alias="ptmPosition", repr=False)
    @property
    def __position_ptm(self) -> int | None:
        """Expose protein modification `position` for serialization.

        Returns:
            out (int | None): `position` when `scope` is `Entity.PROTEIN`,
                otherwise `None`.

        """
        return self.position if self.scope == Entity.PROTEIN else None

    @computed_field(alias="modificationType", repr=False)
    @property
    def __type_base(self) -> str | None:
        """Expose nucleic-acid modification `type` for serialization.

        Returns:
            out (str | None): `type` when `scope` is `Entity.DNA` or
                `Entity.RNA`, otherwise `None`.

        """
        return self.type if self.scope in (Entity.RNA, Entity.DNA) else None

    @computed_field(alias="basePosition", repr=False)
    @property
    def __position_base(self) -> int | None:
        """Expose nucleic-acid modification `position` for serialization.

        Returns:
            out (int | None): `position` when `scope` is `Entity.DNA` or
                `Entity.RNA`, otherwise `None`.

        """
        return (
            self.position if self.scope in (Entity.RNA, Entity.DNA) else None
        )

    @model_serializer(mode="wrap")
    def __scoped_serialization(
        self: Self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Ensure `scope`-dependent serialization of modification.

        This method acts as a fail-safe to ensure that `scope` has been
        assigned prior to serialization. In normal usage, `scope` is set
        automatically by the parent entity (e.g., `DNA`, `RNA`, or `Protein`).

        Raises:
            ValueError: If `scope` is unset at serialization time.

        """
        if self.scope is None:
            msg: str = (
                "Invalid modification: `scope` must be set for serialization."
            )
            raise ValueError(msg)
        return handler(self)
