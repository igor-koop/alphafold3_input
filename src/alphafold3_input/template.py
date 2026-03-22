"""Structural template models.

This submodule defines :class:`Template`, which represents a single
structural template entry for a protein chain in an AlphaFold 3 input.

Exports:
    - :class:`Template`: Structural template entry model.
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
    computed_field,
    model_validator,
)

__all__: list[str] = [
    "Template",
]


class Template(BaseModel):
    """Structural template specification.

    A template is defined by an mmCIF :attr:`structure`, provided either
    inline as a string or as a filesystem path, together with a 0-based
    mapping between query and template residue indexes.

    Attributes:
        structure (str | Path): Template structure.
        indexes (dict[int, int]): Query-to-template residue index mapping.

    Examples:
        Template provided by path.

        >>> Template(
        ...     structure=Path("template.cif.gz"),
        ...     indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
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

    structure: str | Path = Field(
        title="structure",
        alias="structure",
        description="Template structure.",
        validation_alias=AliasChoices("mmcif", "mmcifPath"),
        exclude=True,
    )
    """Template structure."""

    indexes: Mapping[
        Annotated[int, Field(ge=0)],
        Annotated[int, Field(ge=0)],
    ] = Field(
        title="indexes",
        alias="indexes",
        description="Query-to-template residue index mapping.",
        exclude=True,
    )
    """Query-to-template residue index mapping."""

    @computed_field(alias="mmcif", repr=False)
    @property
    def __structure_inline(self) -> str | None:
        """Expose inline ``structure`` for serialization.

        Returns:
            str | None: ``structure`` when it is a string, otherwise ``None``.
        """
        return self.structure if isinstance(self.structure, str) else None

    @computed_field(alias="mmcifPath", repr=False)
    @property
    def __structure_path(self) -> Path | None:
        """Expose path-based ``structure`` for serialization.

        Returns:
            Path | None: ``structure`` when it is a path, otherwise ``None``.
        """
        return self.structure if isinstance(self.structure, Path) else None

    @computed_field(alias="queryIndices", repr=False)
    @property
    def __mapping_query(self) -> tuple[int, ...]:
        """Expose query residue indexes for serialization.

        Returns:
            tuple[int, ...]: Query residue indexes.
        """
        items: tuple[tuple[int, int], ...] = tuple(
            sorted(self.indexes.items(), key=lambda pair: pair[0]),
        )
        return tuple(query for query, _ in items)

    @computed_field(alias="templateIndices", repr=False)
    @property
    def __mapping_template(self) -> tuple[int, ...]:
        """Expose template residue indexes for serialization.

        Returns:
            tuple[int, ...]: Template residue indexes.
        """
        items: tuple[tuple[int, int], ...] = tuple(
            sorted(self.indexes.items(), key=lambda pair: pair[0]),
        )
        return tuple(template for _, template in items)

    @model_validator(mode="wrap")
    @classmethod
    def __coerce_structure(
        cls: type[Self],
        data: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Coerce path-based ``structure`` input to :class:`Path`.

        Args:
            data (Any): Raw input data.
            handler (ModelWrapValidatorHandler[Self]): Inner model validator.

        Returns:
            Self: Validated model with path-based ``structure`` coerced to
            :class:`Path` when applicable.
        """
        if not isinstance(data, Mapping):
            return handler(data)

        alias: str = "mmcifPath"
        value: str | None = data.get(alias)

        model: Self = handler(data)

        if value is not None and isinstance(value, str):
            object.__setattr__(model, "structure", Path(value))

        return model

    @model_validator(mode="before")
    @classmethod
    def __extract_mapping(cls: type[Self], data: Any) -> Any:
        """Extract residue index mapping from query and template indices.

        When the input provides ``queryIndices`` and ``templateIndices``,
        they are replaced by a single :attr:`indexes` mapping.

        Args:
            data (Any): Raw input data.

        Returns:
            Any: Updated input containing :attr:`indexes`, or the original
            input when no extraction is needed.

        Raises:
            ValueError: If the index fields are missing or not sequences of
            integers.
            ValueError: If the two index sequences have different lengths.
        """
        if not isinstance(data, Mapping) or "indexes" in data:
            return data

        out: dict[str, Any] = dict(data)

        query: Sequence[int] | Any = out.pop("queryIndices", None)
        template: Sequence[int] | Any = out.pop("templateIndices", None)

        if (
            query is None
            or not isinstance(query, Sequence)
            or not all(isinstance(i, int) for i in query)
            or template is None
            or not isinstance(template, Sequence)
            or not all(isinstance(i, int) for i in template)
        ):
            msg: str = (
                "Invalid template indexes: both `queryIndices` and "
                "`templateIndices` must be provided as sequences of integer "
                "indexes."
            )
            raise ValueError(msg)

        try:
            indexes: dict[int, int] = dict(zip(query, template, strict=True))
        except ValueError as e:
            msg = (
                "Invalid template indexes: `queryIndices` and "
                "`templateIndices` must have the same length."
            )
            raise ValueError(msg) from e
        out["indexes"] = indexes

        return out
