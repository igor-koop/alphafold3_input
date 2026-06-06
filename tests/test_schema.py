"""Tests for the JSON Schema artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

from alphafold3_input.job import JSON_SCHEMA_URL


class TestSchema:
    """Tests for the JSON Schema artifact."""

    def test_load(self: Self) -> None:
        """Validate that the JSON Schema artifact is valid JSON."""
        path: Path = Path(__file__).parents[1] / "alphafold3-input.schema.json"

        schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

        assert (
            schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        )
        assert schema["$id"] == JSON_SCHEMA_URL
        assert schema["type"] == "object"
        assert "$schema" in schema["properties"]
        assert "sequences" in schema["required"]

    def test_surface(self: Self) -> None:
        """Validate key AlphaFold 3 JSON surface constraints."""
        path: Path = Path(__file__).parents[1] / "alphafold3-input.schema.json"
        schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        defs: dict[str, Any] = schema["$defs"]

        assert schema["properties"]["dialect"]["const"] == "alphafold3"
        assert schema["properties"]["version"]["enum"] == [1, 2, 3, 4]
        assert defs["protein"]["properties"]["sequence"]["pattern"] == (
            "^[ACDEFGHIKLMNPQRSTVWY]+$"
        )
        assert defs["rna"]["properties"]["sequence"]["pattern"] == "^[ACGU]+$"
        assert defs["dna"]["properties"]["sequence"]["pattern"] == "^[ACGT]+$"
        assert defs["ligand"]["oneOf"] == [
            {"required": ["ccdCodes"]},
            {"required": ["smiles"]},
        ]
        assert defs["atom"]["prefixItems"][0] == {"$ref": "#/$defs/chainId"}

    def test_constraints(self: Self) -> None:
        """Validate version-gated and mutually exclusive schema rules."""
        path: Path = Path(__file__).parents[1] / "alphafold3-input.schema.json"
        schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        defs: dict[str, Any] = schema["$defs"]

        assert schema["not"] == {"required": ["userCCD", "userCCDPath"]}
        assert len(schema["allOf"]) == 3

        assert schema["allOf"][0]["if"]["properties"]["version"] == {
            "const": 1,
        }
        assert schema["allOf"][0]["then"]["properties"]["sequences"][
            "items"
        ] == {
            "$ref": "#/$defs/version1Sequence",
        }
        assert schema["allOf"][1]["if"]["properties"]["version"] == {
            "enum": [1, 2],
        }
        assert schema["allOf"][1]["then"] == {
            "not": {"required": ["userCCDPath"]},
        }
        assert schema["allOf"][2]["if"]["properties"]["version"] == {
            "enum": [1, 2, 3],
        }
        assert schema["allOf"][2]["then"]["properties"]["sequences"][
            "items"
        ] == {
            "$ref": "#/$defs/preVersion4Sequence",
        }

        protein_rules: list[dict[str, Any]] = defs["version1Sequence"]["allOf"][
            0
        ]["then"]["properties"]["protein"]["allOf"]
        assert {"not": {"required": ["unpairedMsaPath"]}} in protein_rules
        assert {"not": {"required": ["pairedMsaPath"]}} in protein_rules
        assert defs["template"]["oneOf"] == [
            {"required": ["mmcif"]},
            {"required": ["mmcifPath"]},
        ]
