"""Package metadata for AlphaFold 3 input models.

Provides a normalized interface to distribution metadata for the package.

Exports:
    - :data:`__title__`: Package title.
    - :data:`__description__`: Package summary.
    - :data:`__author__`: Package author.
    - :data:`__version__`: Installed version.
    - :data:`__package__`: Distribution name.
    - :data:`__module_name__`: Import path.
    - :data:`__repository__`: Repository URL.
    - :data:`__documentation__`: Documentation URL.
    - :data:`__issues__`: Issue tracker URL.
"""

from importlib.metadata import PackageMetadata, metadata, version

__all__: list[str] = [
    "__author__",
    "__description__",
    "__documentation__",
    "__issues__",
    "__module_name__",
    "__package__",
    "__repository__",
    "__title__",
    "__version__",
]

__package__: str = "alphafold3_input"
"""Distribution package name."""

__module_name__: str = "alphafold3_input"
"""Importable top-level module name."""

__version__: str = version(__package__)
"""Installed package version string."""

__metadata__: PackageMetadata = metadata(__package__)
"""Raw distribution metadata."""

__title__: str = __metadata__["Name"]
"""Distribution package name."""

__description__: str = __metadata__.get("Summary", "")
"""Package summary."""

__author__: str = __metadata__.get("Author", "")
"""Package author."""

__repository__: str = ""
"""Repository URL."""

__documentation__: str = ""
"""Documentation URL."""

__issues__: str = ""
"""Issue tracker URL."""

for item in __metadata__.json.get("project_url", []):
    label, url = item.split(", ", 1)
    match label:
        case "Repository":
            __repository__ = url
        case "Documentation":
            __documentation__ = url
        case "Issues":
            __issues__ = url
