"""Structural template model.

This submodule defines the `Template` model, which represents a single
structural template entry for a protein chain in an AlphaFold 3 input
for structure prediction.

Exports:
    Template: Model representing a structural template entry.
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

    Represents a single protein structural template for inclusion under
    `Protein.templates` in an AlphaFold 3 input.

    A template is defined by a mmCIF `structure` either provided inline as a
    string or as a filesystem path, together with a 0-based mapping between
    query and template residue indexes.

    Attributes:
        structure (str | Path): Template structure.
        indexes (dict[int, int]): Query-to-template residue index mapping.

    Examples:
        Template provided by path.
        ```python
        Template(
            structure=Path("template.cif.gz"),
            indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
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

    structure: Annotated[
        str | Path,
        Field(
            title="structure",
            description="Template structure.",
            validation_alias=AliasChoices("mmcif", "mmcifPath"),
            exclude=True,
        ),
    ]

    indexes: Annotated[
        Mapping[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]],
        Field(
            title="indexes",
            description="Query-to-template residue index mapping.",
            validation_alias=None,
            exclude=True,
        ),
    ]

    @computed_field(alias="mmcif", repr=False)
    @property
    def __structure_inline(self) -> str | None:
        """Expose inline `structure` for serialization.

        Returns:
            out (str | None): `structure` when it is a string, otherwise
                `None`.

        """
        return self.structure if isinstance(self.structure, str) else None

    @computed_field(alias="mmcifPath", repr=False)
    @property
    def __structure_path(self) -> Path | None:
        """Expose `structure` path for serialization.

        Returns:
            out (Path | None): `structure` when it is a path, otherwise
                `None`.

        """
        return self.structure if isinstance(self.structure, Path) else None

    @computed_field(alias="queryIndices", repr=False)
    @property
    def __mapping_query(self) -> tuple[int, ...]:
        """Expose query residue indexes from `indexes` for serialization.

        Returns:
            out (tuple[int, ...]): `indexes` keys.

        """
        items: tuple[tuple[int, int], ...] = tuple(
            sorted(self.indexes.items(), key=lambda pair: pair[0]),
        )
        return tuple(query for query, _ in items)

    @computed_field(alias="templateIndices", repr=False)
    @property
    def __mapping_template(self) -> tuple[int, ...]:
        """Expose template residue indexes from `indexes` for serialization.

        Returns:
            out (tuple[int, ...]): `indexes` values.

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
        """Coerce `structure` to a `Path`.

        Inspects the raw input data. When the input is a mapping containing
        `mmcifPath`, its value is converted to a `Path` and assigned to
        `structure` after successful model validation.

        Args:
            data (Any): Raw input data.
            handler (ModelWrapValidatorHandler[Self]): Inner model validator.

        Returns:
            out (Self): Validated model with `structure` coerced to `Path` when
                applicable.

        """
        if not isinstance(data, Mapping):
            return handler(data)

        alias: str = "mmcifPath"
        value: object | None = data.get(alias)
        model: Self = handler(data)
        if value is not None and isinstance(value, str):
            object.__setattr__(model, "structure", Path(value))
        return model

    @model_validator(mode="before")
    @classmethod
    def __extract_mapping(cls: type[Self], data: Any) -> Any:
        """Extract residue `indexes`.

        Inspects the raw input data. When the input is a mapping containing
        `queryIndices` and `templateIndices`, both are removed and replaced by
        a single `indexes` field constructed from the paired indices.

        Args:
            data (Any): Raw input data.

        Returns:
            out (Any): Updated input mapping containing `indexes`, or the
                original input when `data` is not a mapping.

        Raises:
            ValueError: If `queryIndices` or `templateIndices` are missing, or
                if either value is not a sequence.

        """
        if not isinstance(data, Mapping) or "indexes" in data:
            return data

        out: dict[str, Any] = dict(data)

        query = out.pop("queryIndices", None)
        template = out.pop("templateIndices", None)

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

        query: Sequence[int]
        template: Sequence[int]

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
