"""Ligand entity models.

This submodule defines :class:`Ligand`, which can be used to include
non-polymeric chemical entities in an AlphaFold 3 input.

Exports:
    - :class:`Ligand`: Ligand entity model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Self

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StringConstraints,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from rdkit import Chem

__all__: list[str] = [
    "Ligand",
]


class Ligand(BaseModel):
    """Ligand entity specification.

    A ligand is defined either by CCD code(s) or by a SMILES string.

    CCD codes (:attr:`definition` as a sequence of CCD codes) are preferred when
    available. Multiple codes can represent composite ligands such as glycans,
    with covalent connectivity specified separately via :attr:`Job.bonds`.
    Custom CCD entries may be provided through :attr:`Job.ccd`.

    A SMILES string (:attr:`definition` as a string) can be used for ligands not
    present in the CCD, but such ligands cannot be referenced in
    :attr:`Job.bonds`.

    Multiple copies of a ligand can be defined either by setting :attr:`copies`
    or by providing multiple explicit identifiers in :attr:`id`. The optional
    :attr:`description` field is supported only when :attr:`Job.version` is set
    to :attr:`Version.IV`.

    Attributes:
        id (str | Sequence[str] | None): Ligand identifier(s).
        description (str | None): Free-text ligand description.
        definition (str | Sequence[str]): Ligand definition as SMILES or CCD
            code(s).
        copies (int): Number of ligand copies.

    Examples:
        Ligand defined by a CCD code.

        >>> Ligand(
        ...     description="Adenosine triphosphate",
        ...     definition=["ATP"],
        ... )

        Multiple copies of an ion.

        >>> Ligand(
        ...     definition=["MG"],
        ...     copies=2,
        ... )

        Custom ligand with explicit identifier defined by SMILES.

        >>> Ligand(
        ...     id=["LIG"],
        ...     description="Aceclidine",
        ...     definition="CC(=O)OC1C[NH+]2CCC1CC2",
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
        description="Ligand identifier(s).",
        min_length=1,
        validation_alias=AliasPath("ligand", "id"),
        serialization_alias="id",
        default=None,
    )
    """Ligand identifier(s)."""

    description: str | None = Field(
        title="description",
        alias="description",
        description="Free-text ligand description.",
        validation_alias=AliasPath("ligand", "description"),
        serialization_alias="description",
        default=None,
    )
    """"Free-text ligand description."""

    definition: (
        Annotated[
            str,
            StringConstraints(
                pattern=r"^[0-9\-=#/\\@$+()%.*:\[\]BCEFHIKLNOPRSTZbceinos]+$",
            ),
        ]
        | Sequence[str]
    ) = Field(
        title="definition",
        alias="definition",
        description="Ligand definition as CCD code(s) or SMILES.",
        validation_alias=AliasChoices(
            AliasPath("ligand", "ccdCodes"),
            AliasPath("ligand", "smiles"),
        ),
        exclude=True,
    )
    """Ligand definition as CCD code(s) or SMILES."""

    copies: int = Field(
        title="copies",
        description="Number of ligand copies.",
        ge=1,
        validation_alias="copies",
        exclude=True,
        repr=False,
        default=1,
    )
    """Number of ligand copies."""

    @computed_field(alias="smiles", repr=False)
    @property
    def __definition_smiles(self: Self) -> str | None:
        """Expose SMILES ``definition`` for serialization.

        Returns:
            str | None: ``definition`` when it is a SMILES string, otherwise
            ``None``.
        """
        return self.definition if isinstance(self.definition, str) else None

    @computed_field(alias="ccdCodes", repr=False)
    @property
    def __definition_ccd(self: Self) -> Sequence[str] | None:
        """Expose CCD code definitions for serialization.

        Returns:
            Sequence[str] | None: ``definition`` when it is a sequence of CCD
            codes, otherwise ``None``.
        """
        return self.definition if not isinstance(self.definition, str) else None

    @model_validator(mode="after")
    def __validate_copies(self: Self) -> Self:
        """Validate consistency between ``id`` and ``copies``.

        If :attr:`id` is provided, :attr:`copies` is set to ``len(id)``. If
        :attr:`copies` was explicitly provided and is inconsistent with
        :attr:`id`, a :class:`ValueError` is raised.

        Returns:
            Self: Validated ligand instance.

        Raises:
            ValueError: If ``copies`` is inconsistent with ``id``.
        """
        if self.id is None:
            return self

        n: int = len(self.id) if not isinstance(self.id, str) else 1

        if "copies" in self.model_fields_set and self.copies != n:
            msg: str = (
                "Conflicting ligand configuration: `copies` is inconsistent "
                f"with the length of `id` (copies={self.copies}, len(id)={n})."
            )
            raise ValueError(msg)

        object.__setattr__(self, "copies", n)
        return self

    @field_validator("definition", mode="after")
    @classmethod
    def __validate_smiles(
        cls: type[Self],
        value: str | Sequence[str],
    ) -> str | Sequence[str]:
        """Validate SMILES ligand definitions.

        If :attr:`definition` is provided as a string, it is interpreted as a
        SMILES string and must be syntactically valid, chemically valid, and in
        canonical form. CCD code definitions are returned unchanged.

        Args:
            value (str | Sequence[str]): Ligand definition to validate.

        Returns:
            str | Sequence[str]: Validated ligand definition.

        Raises:
            ValueError: If the SMILES string is invalid or not canonical.
        """
        if not isinstance(value, str):
            return value

        molecule: Chem.Mol | None = Chem.MolFromSmiles(
            SMILES=value,
            sanitize=False,
        )

        if molecule is None:
            msg: str = (
                "Invalid ligand SMILES: `definition` is not syntactically "
                f"valid (definition={value!r})."
            )
            raise ValueError(msg)

        try:
            Chem.SanitizeMol(mol=molecule)
        except Exception as e:
            msg: str = (
                "Invalid ligand SMILES: `definition` does not describe a "
                f"chemically valid structure (definition={value!r})."
            )
            raise ValueError(msg) from e

        canonical: str = Chem.MolToSmiles(
            mol=molecule,
            canonical=True,
            isomericSmiles=True,
        )

        if value != canonical:
            msg: str = (
                "Invalid ligand SMILES: `definition` is not in canonical "
                f"form (definition={value!r}, canonical={canonical!r})."
            )
            raise ValueError(msg)

        return value

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
