"""Ligand entity model.

This submodule defines the `Ligand` model, which can be used to include
non-polymeric chemical entities, such as small molecules or ions, in an
AlphaFold 3 input for structure prediction.

Exports:
    Ligand: Model representing a ligand entity.
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Annotated, Any, Self

from pydantic import (
    AliasChoices,
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

    Represents a non-polymeric chemical entity (such as a small molecule or
    ion) for inclusion under `Job.sequences` in an AlphaFold 3 input.

    A ligand is defined either by CCD code(s) or a SMILES string:
    - CCD codes (`definition` as `Sequence[str]`) are preferred if available.
      Multiple codes can be used to represent composite ligands (such as
      glycans) with covalent bonds specified via `Job.bonds`, using explicit
      atom names. Custom CCDs can be used when provided via `Job.ccd`.
    - SMILES (`definition` as `str`) allows specifying ligands not present in
      the CCD, but cannot be referenced by `Job.bonds` and may be less robust
      for geometry generation across random seeds.

    Multiple copies of a ligand can be defined either by setting `copies` or by
    providing multiple explicit identifiers via `id`. Optional `description` is
    supported only when `Job.version` is set to input format version 4.


    Attributes:
        id (Sequence[stt] | None): Ligand identifier(s).
        description (str | None): Free-text ligand description.
        definition (str | Sequence[str]): Ligand definition as SMILES or CCD
            code(s).
        copies (int): Number of ligand copies.

    Examples:
        Ligand defined by a CCD code.
        ```python
        Ligand(
            description="Adenosine triphosphate",
            definition=["ATP"],
        )
        ```

        Multiple copies of an ion.
        ```python
        Ligand(
            definition=["MG"],
            copies=2,
        )
        ```

        Custom ligand with explicit identifier defined by SMILES.
        ```python
        Ligand(
            id=["LIG"],
            description="Aceclidine",
            definition="CC(=O)OC1C[NH+]2CCC1CC2",
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
            description="Ligand identifier(s).",
            min_length=1,
            validation_alias="id",
            serialization_alias="id",
        ),
    ] = None

    description: Annotated[
        str | None,
        Field(
            title="description",
            description="Free-text ligand description.",
            validation_alias="description",
            serialization_alias="description",
        ),
    ] = None

    definition: Annotated[
        Annotated[
            str,
            StringConstraints(
                pattern=r"^[0-9\-=#/\\@$+()%.*:\[\]BCEFHIKLNOPRSTZbceinos]+$",
            ),
        ]
        | Sequence[str],
        Field(
            title="definition",
            description="Ligand definition as CCD code(s) or SMILES.",
            validation_alias=AliasChoices("ccdCodes", "smiles"),
            exclude=True,
        ),
    ]

    copies: Annotated[
        int,
        Field(
            title="copies",
            description="Number of ligand copies.",
            ge=1,
            validation_alias="copies",
            exclude=True,
            repr=False,
        ),
    ] = 1

    @computed_field(alias="smiles", repr=False)
    @property
    def __definition_smiles(self: Self) -> str | None:
        """Expose SMILES `definition` for serialization.

        Returns:
            out (str | None): `definition` when it is a SMILES string,
                otherwise `None`.

        """
        return self.definition if isinstance(self.definition, str) else None

    @computed_field(alias="ccdCodes", repr=False)
    @property
    def __definition_ccd(self: Self) -> Sequence[str] | None:
        """Expose CCD code(s) `definition` for serialization.

        Returns:
            out (Sequence[str, ...] | None): `definition` when it is a
                sequence of CCD codes, otherwise `None`.

        """
        return (
            self.definition if not isinstance(self.definition, str) else None
        )

    @model_validator(mode="after")
    def __validate_copies(self: Self) -> Self:
        """Enforce consistency between `id` and `copies`.

        If `id` is provided, `copies` is set to `len(id)`. If `copies` was
        explicitly provided and is inconsistent with `id`, a `ValueError` is
        raised.

        Returns:
            out (Self): The validated ligand instance.

        Raises:
            ValueError: If `copies` was explicitly provided and is inconsistent
                with `id`.

        """
        if self.id is None:
            return self

        n: int = len(self.id)

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
        """Validate SMILES ligand definitions and enforce canonical form.

        If `definition` is provided as a string, it is interpreted as SMILES
        and must be syntactically correct, chemically valid, and in canonical
        form. Otherwise, a `ValueError` is raised. If provided as a sequence of
        CCD codes, it is returned unchanged.

        Args:
            value (str | Sequence[str]): The ligand definition to validate.

        Returns:
            out (str | Sequence[str]): The validated ligand definition.

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
        """Serialize the entity using the AlphaFold 3 wrapped representation.

        Returns:
            out (dict[str, Any]): Wrapped entity mapping suitable for the
                AlphaFold 3 input format.

        """
        return {self.__class__.__name__.lower(): handler(self)}
