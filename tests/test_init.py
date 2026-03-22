"""Tests for the public package namespace."""

from __future__ import annotations

from typing import Any

import alphafold3_input as package


def test_package_docstring() -> None:
    """Validate that the package has a non-empty module docstring."""
    assert isinstance(package.__doc__, str)
    assert package.__doc__.strip()


def test_exports_valid() -> None:
    """Validate that the package exports are public, defined and not None."""
    assert package.__all__
    for name in package.__all__:
        value: Any = getattr(package, name, None)
        assert value is not None, f"Invalid export: {name}"


def test_exports_match() -> None:
    """Validate that the package exports match the expected ones."""
    expected: set[str] = {
        "Atom",
        "Bond",
        "DNA",
        "Dialect",
        "Entity",
        "Job",
        "Ligand",
        "Modification",
        "Operation",
        "Protein",
        "RNA",
        "Template",
        "Version",
        "ccd",
        "component",
        "realign",
        "reindex",
        "trace",
    }
    assert set(package.__all__) == expected
