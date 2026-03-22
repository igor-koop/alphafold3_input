"""Protein chain entity models.

This submodule defines :class:`Protein`, which can be used to include
polymeric protein entities in an AlphaFold 3 input.

Exports:
    - :class:`Protein`: Protein chain entity model.
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
from .template import Template

__all__: list[str] = [
    "Protein",
]


class Protein(BaseModel):
    """Protein chain entity specification.

    A protein chain is defined by its amino acid :attr:`sequence`, optional
    residue :attr:`modifications`, optional multiple sequence :attr:`alignment`,
    and optional structural :attr:`templates`.

    Post-translational modifications can be provided through
    :class:`Modification` entries. Additional entries may be appended using
    :meth:`modify`.

    A multiple sequence alignment may be provided either inline as a string or
    as a filesystem path. When :attr:`alignment` is present, paired MSA output
    is serialized as an empty string.

    Multiple copies of a protein chain can be defined either by setting
    :attr:`copies` or by providing multiple explicit identifiers in
    :attr:`id`. The optional :attr:`description` field is supported only when
    :attr:`Job.version` is set to :attr:`Version.IV`.

    Attributes:
        id (str | Sequence[str] | None): Protein chain identifier(s).
        description (str | None): Free-text protein chain description.
        sequence (str): Protein chain amino acid sequence.
        modifications (Sequence[Modification]): Protein chain
            post-translational modifications.
        alignment (str | Path | None): Multiple sequence alignment.
        templates (Sequence[Template] | None): Structural templates.
        copies (int): Number of protein chain copies.

    Examples:
        Protein chain with a description.

        >>> Protein(
        ...     description="AviTag for BirA-mediated biotinylation",
        ...     sequence="GLNDIFEAQKIEWHE",
        ... )

        Multiple copies of a protein chain.

        >>> Protein(
        ...     sequence="RMKQLEDKVEELLSKKYHLENEVARLKKLVGER",
        ...     copies=2,
        ... )

        Protein chain with modified residues.

        >>> peptide = Protein(sequence="PVLSCGEWQL")
        >>> peptide.modify(
        ...     Modification(type="HY3", position=1),
        ...     Modification(type="P1L", position=5),
        ... )

        Protein chain with an alignment file.

        >>> Protein(
        ...     id=["A", "B"],
        ...     sequence="KRRWKKNFIAVSAANRFKKISSSGAL",
        ...     alignment=Path("alignment.a3m"),
        ... )

        Protein chain with a structural template.

        >>> Protein(
        ...     id=["C"],
        ...     sequence="RPACQLW",
        ...     templates=[
        ...         Template(
        ...             structure=Path("template.cif.gz"),
        ...             indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
        ...         ),
        ...     ],
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
        description="Protein chain identifier(s).",
        min_length=1,
        validation_alias=AliasPath("protein", "id"),
        serialization_alias="id",
        default=None,
    )
    """Protein chain identifier(s)."""

    description: str | None = Field(
        title="description",
        alias="description",
        description="Free-text protein chain description.",
        validation_alias=AliasPath("protein", "description"),
        serialization_alias="description",
        default=None,
    )
    """Free-text protein chain description."""

    sequence: Annotated[
        str,
        StringConstraints(pattern="^[ACDEFGHIKLMNPQRSTVWY]+$"),
    ] = Field(
        title="sequence",
        alias="sequence",
        description="Protein chain amino acid sequence.",
        min_length=1,
        validation_alias=AliasPath("protein", "sequence"),
        serialization_alias="sequence",
    )
    """Protein chain amino acid sequence."""

    modifications: Sequence[Modification] = Field(
        title="modifications",
        alias="modifications",
        description="Protein chain residue modifications.",
        validation_alias=AliasPath("protein", "modifications"),
        serialization_alias="modifications",
        default_factory=tuple,
    )
    """Protein chain residue modifications."""

    alignment: str | Path | None = Field(
        title="alignment",
        alias="alignment",
        description="Multiple sequence alignment for protein chain.",
        validation_alias=AliasChoices(
            AliasPath("protein", "unpairedMsa"),
            AliasPath("protein", "unpairedMsaPath"),
        ),
        exclude=True,
        default=None,
    )
    """Multiple sequence alignment for protein chain."""

    templates: Sequence[Template] | None = Field(
        title="templates",
        alias="templates",
        description="Structural templates for protein chain.",
        validation_alias=AliasPath("protein", "templates"),
        serialization_alias="templates",
        default=None,
    )
    """Structural templates for protein chain."""

    copies: int = Field(
        title="copies",
        alias="copies",
        description="Number of protein chain copies.",
        ge=1,
        validation_alias="copies",
        exclude=True,
        repr=False,
        default=1,
    )
    """Number of protein chain copies."""

    @computed_field(alias="unpairedMsa", repr=False)
    @property
    def __alignment_inline(self: Self) -> str | None:
        """Expose inline ``alignment`` for serialization.

        Returns:
            str | None: ``alignment`` when it is a string, otherwise ``None``.
        """
        return self.alignment if isinstance(self.alignment, str) else None

    @computed_field(alias="unpairedMsaPath", repr=False)
    @property
    def __alignment_path(self: Self) -> Path | None:
        """Expose path-based ``alignment`` for serialization.

        Returns:
            Path | None: ``alignment`` when it is a path, otherwise ``None``.
        """
        return self.alignment if isinstance(self.alignment, Path) else None

    @computed_field(alias="pairedMsa", repr=False)
    @property
    def __alignment_paired(self: Self) -> str | None:
        """Expose paired MSA output for serialization.

        Returns:
            str | None: Empty string when :attr:`alignment` is present,
            otherwise ``None``.
        """
        return "" if self.alignment is not None else None

    def modify(self: Self, *modifications: Modification) -> Self:
        """Append residue modifications to the protein chain.

        Args:
            *modifications (Modification): One or more modifications to add.

        Returns:
            Self: Protein chain with appended modifications.

        Raises:
            TypeError: If no modifications were provided.
        """
        if not modifications:
            msg: str = (
                "Invalid protein modification: no modifications were provided."
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

        if isinstance(data.get("protein"), Mapping):
            value: str | None = data["protein"].get(alias)
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
        """Assign protein scope to each modification.

        Args:
            value (Sequence[Modification]): Modification entries.

        Returns:
            Sequence[Modification]: Scoped modification entries.
        """
        for modification in value:
            object.__setattr__(modification, "scope", Entity.PROTEIN)
        return value

    @model_validator(mode="after")
    def __validate_modifications(self: Self) -> Self:
        """Validate residue modifications against the protein sequence.

        Ensures that each modification position is within the sequence and that
        no residue is modified more than once.

        Returns:
            Self: Validated protein chain instance.

        Raises:
            ValueError: If a modification position is out of range.
            ValueError: If multiple modifications target the same position.
        """
        length: int = len(self.sequence)
        modified: set[int] = set()

        for modification in self.modifications:
            if modification.position > length:
                msg: str = (
                    "Invalid protein modification: modification `position` is "
                    f"out of range (position={modification.position}, "
                    f"len(sequence)={length})."
                )
                raise ValueError(msg)

            if modification.position in modified:
                msg: str = (
                    "Invalid protein modification list: multiple "
                    "modifications on the same `position` "
                    f"(position={modification.position})."
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
            Self: Validated protein chain instance.

        Raises:
            ValueError: If ``copies`` is inconsistent with ``id``.
        """
        if self.id is None:
            return self

        n: int = len(self.id) if not isinstance(self.id, str) else 1

        if "copies" in self.model_fields_set and self.copies != n:
            msg: str = (
                "Conflicting protein configuration: `copies` is inconsistent "
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
