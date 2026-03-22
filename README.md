# alphafold3_input

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://igor-koop.github.io/alphafold3_input/)
[![License](https://img.shields.io/github/license/igor-koop/alphafold3_input)](https://github.com/igor-koop/alphafold3_input/blob/main/LICENSE)
[![Coverage](https://codecov.io/gh/igor-koop/alphafold3_input/branch/main/graph/badge.svg)](https://codecov.io/gh/igor-koop/alphafold3_input)
[![Lint](https://img.shields.io/badge/linting-ruff-d7ff64)](https://docs.astral.sh/ruff/)
[![Type](https://img.shields.io/badge/typing-ty-7c3aed)](https://docs.astral.sh/ty/)
[![Ations](https://github.com/igor-koop/alphafold3_input/actions/workflows/ci-pages.yml/badge.svg)](https://github.com/igor-koop/alphafold3_input/actions/workflows/ci-pages.yml)

Pythonic, object-oriented models for constructing **AlphaFold 3** input files.

This package provides typed models and utilities that abstract the AlphaFold 3
JSON input format into a clean, validated Python interface.

For details on the underlying specification, see the official
[AlphaFold 3 input specification](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md).

---

## Installation

```bash
uv add git+https://github.com/igor-koop/alphafold3_input
```

## Documentation

Full API documentation is available at: https://igor-koop.github.io/alphafold3_input/

## Development

```bash
git clone https://github.com/igor-koop/alphafold3_input
cd alphafold3_input

uv sync --dev
```

## License

The scripts and documentation in this project are released under the [MIT License](https://github.com/igor-koop/alphafold3_input/blob/main/LICENSE).