"""Covalent bond model.

This submodule defines `Atom` and `Bond` models for specifying covalent bonds
between entities in an AlphaFold 3 input.

Exports:
    Atom: Model representing an atom specification within an entity.
    Bond: Model representing a covalent bond between two atoms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

__all__: list[str] = [
    "Atom",
    "Bond",
]


class Atom(BaseModel):
    """Atom specification within an entity.

    Represents an atom in a polymer or ligand entity using `entity` identifier,
    1-based `residue` index, and atom `name`.

    Attributes:
        entity (str): Entity identifier.
        residue (int): Residue index within the entity.
        name (str): Atom name within the residue.

    Examples:
        Atom definition.
        ```python
        Atom(
            entity="A",
            residue=1,
            name="CB",
        )
        ```

    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=False,
        validate_by_name=True,
        validate_by_alias=True,
    )

    entity: Annotated[
        str,
        Field(
            title="entity",
            description="Entity identifier.",
            pattern="^[A-Z]+$",
            validation_alias="entity",
            serialization_alias="entity",
        ),
    ]

    residue: Annotated[
        int,
        Field(
            title="residue",
            description="Residue index within the entity.",
            ge=1,
            validation_alias="residue",
            serialization_alias="residue",
        ),
    ]

    name: Annotated[
        str,
        Field(
            title="name",
            description="Atom name within the residue.",
            min_length=1,
            validation_alias="name",
            serialization_alias="name",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def __validate_model(cls: type[Self], data: Any) -> Any:
        """Coerce compact atom definitions to a mapping.

        Accepts the AlphaFold 3 list form `[entity, residue, name]` and
        converts it to a mapping compatible with field validation.

        Args:
            data (Any): Raw input data.

        Returns:
            out (Any): Mapping with keys `entity`, `residue`, and `name` when
                `data` is a sequence, otherwise the original input.

        Raises:
            ValueError: If `data` is a sequence but does not have exactly three
                items (`[entity, residue, name]`).

        """
        if not isinstance(data, Sequence) or isinstance(
            data,
            (str, bytes, bytearray),
        ):
            return data

        if len(data) != len(cls.model_fields):
            msg: str = (
                "Invalid atom definition: expected `[entity, residue, name]`."
            )
            raise ValueError(msg)
        return {"entity": data[0], "residue": data[1], "name": data[2]}

    @model_serializer(mode="plain")
    def __serialize_model(self: Self) -> tuple[str, int, str]:
        """Serialize atom as `[entity, residue, name]`.

        Returns:
            out (tuple[str, int, str]): Compact AlphaFold 3 atom
                representation.

        """
        return (self.entity, self.residue, self.name)


class Bond(BaseModel):
    """Covalent bond specification.

    Represents a single covalent bond between two atoms, encoded as an
    AlphaFold 3 bonded atom pair.

    Bonds are intended for covalently-linked ligands and multi-residue ligand
    entities (e.g. glycans). Defining covalent bonds between or within polymer
    entities (DNA, RNA, or protein) is not supported by AlphaFold 3.

    Attributes:
        source (Atom): Source atom address.
        target (Atom): Target atom address.

    Examples:
        Covalent bond between two entities.
        ```python
        Bond(
            source=Atom(entity="A", residue=1, name="CA"),
            target=Atom(entity="G", residue=1, name="CHA"),
        )
        ```

        Covalent bond within a multi-residue entity.
        ```python
        Bond(
            source=Atom(entity="I", residue=1, name="O6"),
            target=Atom(entity="I", residue=2, name="C1"),
        )
        ```

    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    source: Annotated[
        Atom,
        Field(
            title="source",
            description="Source atom address.",
        ),
    ]

    target: Annotated[
        Atom,
        Field(
            title="target",
            description="Target atom address.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def __validate_model(cls: type[Self], data: Any) -> Any:
        """Coerce compact bond definitions to a mapping.

        Accepts the AlphaFold 3 list form `[[...], [...]]` (a bonded atom pair)
        and converts it to a mapping compatible with field validation.

        Args:
            data (Any): Raw input data.

        Returns:
            out (Any): Mapping with keys `source` and `target` when `data` is a
                sequence, otherwise the original input.

        Raises:
            ValueError: If `data` is a sequence but does not have exactly two
                items (`[source, target]`).

        """
        if not isinstance(data, Sequence) or isinstance(
            data,
            (str, bytes, bytearray),
        ):
            return data

        if len(data) != len(cls.model_fields):
            msg: str = "Invalid bond definition: expected `[source, target]`."
            raise ValueError(msg)
        return {"source": data[0], "target": data[1]}

    @model_serializer(mode="plain")
    def __serialize_model(
        self: Self,
    ) -> tuple[Atom, Atom]:
        """Serialize bond as an AlphaFold 3 bonded atom pair.

        Returns:
            out (tuple[Atom, Atom]): Compact AlphaFold 3 bonded atom pair
                representation.

        """
        return (self.source, self.target)
