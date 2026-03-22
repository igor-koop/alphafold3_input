"""Covalent bond models.

This submodule defines :class:`Atom` and :class:`Bond` for specifying
covalent bonds between entities in an AlphaFold 3 input.

Exports:
    - :class:`Atom`: atom specification within an entity.
    - :class:`Bond`: covalent bond between two atoms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

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

    An atom is defined by an :attr:`entity` identifier, a 1-based
    :attr:`residue` index, and an atom :attr:`name`.

    Attributes:
        entity (str): Entity identifier.
        residue (int): Residue index within the entity.
        name (str): Atom name within the residue.

    Examples:
        Atom definition.

        >>> Atom(entity="A", residue=1, name="CB")
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=False,
        validate_by_name=True,
        validate_by_alias=True,
    )

    entity: str = Field(
        title="entity",
        alias="entity",
        description="Entity identifier.",
        pattern="^[A-Z]+$",
        validation_alias="entity",
        serialization_alias="entity",
    )
    """Entity identifier."""

    residue: int = Field(
        title="residue",
        alias="residue",
        description="Residue index within the entity.",
        ge=1,
        validation_alias="residue",
        serialization_alias="residue",
    )
    """Residue index within the entity."""

    name: str = Field(
        title="name",
        alias="name",
        description="Atom name within the residue.",
        min_length=1,
        validation_alias="name",
        serialization_alias="name",
    )
    """Atom name within the residue."""

    @model_validator(mode="before")
    @classmethod
    def __validate_model(cls: type[Self], data: Any) -> Any:
        """Coerce compact atom definitions to a mapping.

        Accepts the AlphaFold 3 list form ``[entity, residue, name]`` and
        converts it to a mapping compatible with field validation.

        Args:
            data (Any): Raw input data.

        Returns:
            Any: A mapping with keys ``entity``, ``residue``, and ``name`` if
                ``data`` is a sequence, otherwise the original input.

        Raises:
            ValueError: If ``data`` is a sequence but does not contain exactly
                three items.
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
        """Serialize the atom as ``[entity, residue, name]``.

        Returns:
            tuple[str, int, str]: Compact AlphaFold 3 atom representation.
        """
        return (self.entity, self.residue, self.name)


class Bond(BaseModel):
    """Covalent bond specification.

    Defines a covalent bond between the :attr:`source` and the :attr:`target`
    atoms as an AlphaFold 3 bonded atom pair.

    Bonds are intended for covalently linked multi-residue :class:`Ligand`
    entities, for example glycans. Covalent bonds within or between polymer
    entities such as :class:`DNA`, :class:`RNA`, or :class:`Protein` are not
    supported by AlphaFold 3.

    Attributes:
        source (Atom): Source atom address.
        target (Atom): Target atom address.

    Examples:
        Covalent bond between two entities.

        >>> Bond(
        ...     source=Atom(entity="A", residue=1, name="CA"),
        ...     target=Atom(entity="G", residue=1, name="CHA"),
        ... )

        Covalent bond within a multi-residue entity.

        >>> Bond(
        ...     source=Atom(entity="I", residue=1, name="O6"),
        ...     target=Atom(entity="I", residue=2, name="C1"),
        ... )
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    source: Atom = Field(
        title="source",
        alias="source",
        description="Source atom address.",
        validation_alias="source",
        serialization_alias="source",
    )
    """Source atom address."""

    target: Atom = Field(
        title="target",
        alias="target",
        description="Target atom address.",
        validation_alias="target",
        serialization_alias="target",
    )
    """Target atom address."""

    @model_validator(mode="before")
    @classmethod
    def __validate_model(cls: type[Self], data: Any) -> Any:
        """Coerce compact bond definitions to a mapping.

        Accepts the AlphaFold 3 list form ``[[...], [...]]`` and converts it
        to a mapping compatible with field validation.

        Args:
            data (Any): Raw input data.

        Returns:
            Any: A mapping with keys ``source`` and ``target`` if ``data`` is
                a sequence, otherwise the original input.

        Raises:
            ValueError: If ``data`` is a sequence but does not contain exactly
                two items.
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
    def __serialize_model(self: Self) -> tuple[Atom, Atom]:
        """Serialize the bond as an AlphaFold 3 bonded atom pair.

        Returns:
            tuple[Atom, Atom]: Compact AlphaFold 3 bonded atom pair
                representation.
        """
        return (self.source, self.target)
