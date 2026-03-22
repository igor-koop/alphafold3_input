"""Tests for structural template models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest
from pydantic import ValidationError

from alphafold3_input.template import Template

if TYPE_CHECKING:
    from pathlib import Path


class TestTemplate:
    """Tests for the ``Template`` model."""

    def test_construct_mapping(self: Self) -> None:
        """Validate construction from a mapping."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8},
        )

        assert template.structure == "data_template"
        assert template.indexes == {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8}

    def test_construct_alias_inline(self: Self) -> None:
        """Validate construction from inline mmCIF alias input."""
        template: Template = Template.model_validate(
            {
                "mmcif": "data_template",
                "queryIndices": [0, 1, 2, 4, 5, 6],
                "templateIndices": [0, 1, 2, 3, 4, 8],
            },
        )

        assert template.structure == "data_template"
        assert template.indexes == {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 8}

    def test_construct_alias_path(self: Self, tmp_path: Path) -> None:
        """Validate construction from path-based mmCIF alias input."""
        path: Path = tmp_path / "template.cif"

        template: Template = Template.model_validate(
            {
                "mmcifPath": str(path),
                "queryIndices": [0, 1, 2],
                "templateIndices": [3, 4, 5],
            },
        )

        assert template.structure == path
        assert template.indexes == {0: 3, 1: 4, 2: 5}

    def test_validate_indexes_extracted(self: Self) -> None:
        """Validate extraction of residue index mapping."""
        template: Template = Template.model_validate(
            {
                "mmcif": "data_template",
                "queryIndices": [0, 1, 2],
                "templateIndices": [3, 4, 5],
            },
        )

        assert template.indexes == {0: 3, 1: 4, 2: 5}

    def test_validate_indexes_passthrough(self: Self) -> None:
        """Validate that explicit index mappings are accepted unchanged."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 0, 2: 5, 4: 8},
        )

        assert template.indexes == {0: 0, 2: 5, 4: 8}

    def test_validate_indexes_missing(self: Self) -> None:
        """Validate that both index sequences are required for extraction."""
        with pytest.raises(
            ValidationError,
            match="both `queryIndices` and `templateIndices`",
        ):
            Template.model_validate(
                {
                    "mmcif": "data_template",
                    "queryIndices": [0, 1, 2],
                },
            )

    def test_validate_indexes_type(self: Self) -> None:
        """Validate that extracted index sequences must contain integers."""
        with pytest.raises(
            ValidationError,
            match="both `queryIndices` and `templateIndices`",
        ):
            Template.model_validate(
                {
                    "mmcif": "data_template",
                    "queryIndices": [0, 1, "2"],
                    "templateIndices": [3, 4, 5],
                },
            )

    def test_validate_indexes_length(self: Self) -> None:
        """Validate that extracted index sequences must have equal length."""
        with pytest.raises(ValidationError, match="must have the same length"):
            Template.model_validate(
                {
                    "mmcif": "data_template",
                    "queryIndices": [0, 1, 2],
                    "templateIndices": [3, 4],
                },
            )

    @pytest.mark.parametrize(
        "indexes",
        [
            {-1: 0},
            {0: -1},
            {-1: -1},
        ],
    )
    def test_validate_indexes_nonnegative(
        self: Self,
        indexes: dict[int, int],
    ) -> None:
        """Validate that query and template indexes must be non-negative."""
        with pytest.raises(ValidationError):
            Template(
                structure="data_template",
                indexes=indexes,
            )

    def test_serialize_inline(self: Self) -> None:
        """Validate serialization of inline mmCIF structures."""
        template: Template = Template(
            structure="data_template",
            indexes={0: 0, 1: 1, 2: 2},
        )

        data: dict[str, object] = template.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert data["mmcif"] == "data_template"
        assert data["queryIndices"] == (0, 1, 2)
        assert data["templateIndices"] == (0, 1, 2)
        assert "mmcifPath" not in data
        assert "structure" not in data
        assert "indexes" not in data

    def test_serialize_path(self: Self, tmp_path: Path) -> None:
        """Validate serialization of path-based mmCIF structures."""
        path: Path = tmp_path / "template.cif"

        template: Template = Template(
            structure=path,
            indexes={0: 3, 1: 4, 2: 5},
        )

        data: dict[str, object] = template.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert data["mmcifPath"] == path
        assert data["queryIndices"] == (0, 1, 2)
        assert data["templateIndices"] == (3, 4, 5)
        assert "mmcif" not in data
        assert "structure" not in data
        assert "indexes" not in data

    def test_serialize_indexes(self: Self) -> None:
        """Validate serialization of indexes and their sorted order."""
        template: Template = Template(
            structure="data_template",
            indexes={4: 8, 0: 0, 2: 5},
        )

        data: dict[str, object] = template.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        assert data["queryIndices"] == (0, 2, 4)
        assert data["templateIndices"] == (0, 5, 8)
