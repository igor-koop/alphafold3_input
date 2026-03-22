"""AlphaFold 3 input models.

This package provides models for constructing AlphaFold 3 input files.

It offers a Pythonic, object-oriented interface for defining AlphaFold 3
jobs, abstracting the underlying JSON input format into typed models and
validated structures. The implementation closely follows the official
AlphaFold 3 input specification provided by DeepMind.

For full details on the expected input format and supported features,
refer to the official `AlphaFold 3 input specification
<https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md>`_.

Exports:
    - :class:`Job`, :class:`Dialect`, :class:`Version`: top-level job model \
        and input format enums.
    - :class:`DNA`, :class:`RNA`, :class:`Protein`, :class:`Ligand`: \
        entity models used under :attr:`Job.entities`.
    - :class:`Modification`, :class:`Entity`: residue modification model and \
        its entity-scope enum.
    - :class:`Template`: structural template specification for proteins.
    - :class:`Atom`, :class:`Bond`: covalent bond specification models.
    - :class:`Operation`, :func:`trace`, :func:`reindex`, :func:`realign`: \
        operation trace generation, reindexing, and realignment utilities.
    - :func:`ccd`, :func:`component`: generation of custom chemical \
        component dictionaries.
"""

from .bond import Atom, Bond
from .dna import DNA
from .job import Dialect, Job, Version
from .ligand import Ligand
from .modification import Entity, Modification
from .protein import Protein
from .rna import RNA
from .template import Template
from .utils import Operation, ccd, component, realign, reindex, trace

__all__: list[str] = [
    "DNA",
    "RNA",
    "Atom",
    "Bond",
    "Dialect",
    "Entity",
    "Job",
    "Ligand",
    "Modification",
    "Operation",
    "Protein",
    "Template",
    "Version",
    "ccd",
    "component",
    "realign",
    "reindex",
    "trace",
]
