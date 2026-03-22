"""Tests for package metadata."""

from __future__ import annotations

from importlib.metadata import metadata, version
from typing import Any

from pydantic import HttpUrl, TypeAdapter

import alphafold3_input.__about__ as about

URL: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


def test_exports_valid() -> None:
    """Validate that metadata values are defined, non-empty, and valid."""
    assert about.__all__
    for name in about.__all__:
        value: Any = getattr(about, name, None)

        assert value is not None, f"Invalid export: {name}"
        assert isinstance(value, str), f"Export not a string: {name}"
        assert value.strip(), f"Export is empty: {name}"

        if name in {"__repository__", "__documentation__", "__issues__"}:
            assert URL.validate_python(value), f"{name} is not a valid URL"


def test_exports_match() -> None:
    """Validate that the metadata exports match the expected ones."""
    expected: set[str] = {
        "__author__",
        "__description__",
        "__documentation__",
        "__issues__",
        "__module_name__",
        "__package__",
        "__repository__",
        "__title__",
        "__version__",
    }
    assert set(about.__all__) == expected


def test_match_names() -> None:
    """Validate that the metadata package and module names match."""
    assert about.__package__ == "alphafold3_input"
    assert about.__module_name__ == "alphafold3_input"


def test_match_distribution() -> None:
    """Validate that the metadata matches the installed distribution."""
    assert about.__version__ == version(about.__package__)
    assert about.__title__ == metadata(about.__package__).get("Name")
    assert about.__description__ == metadata(about.__package__).get("Summary")
