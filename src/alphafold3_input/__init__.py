"""AlphaFold 3 input models.

This package provides models for constructing AlphaFold 3 input files.

Exports:
    Atom, Bond: Covalent bond specification models.
    DNA, RNA, Protein, Ligand: Entity models used under `Job.entities`.
    Modification, Entity: Residue modification model and its entity-scope enum.
    Template: Structural template specification for proteins.
    Job, Dialect, Version: Top-level job model and input format enums.
    Operation, trace, reindex, realign: Operation trace generation, reindexing,
        and realignment.
    ccd, component: Generation of custom chemical component dictionary.
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
