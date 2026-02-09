"""Metadata for AlphaFold 3 input models.

This submodule defines and exposes package metadata such as the version,
repository URL, and documentation URL. This information is extracted from the
package's distribution metadata.

Exports:
    __documentation__: URL to the package documentation.
    __issues__: URL to the package issue tracker.
    __repository__: URL to the package repository.
    __module__: Name of the package.
    __package__: Name of the package.
    __version__: Version of the package.
"""

from importlib.metadata import PackageMetadata, metadata, version

__all__: list[str] = [
    "__documentation__",
    "__issues__",
    "__module__",
    "__package__",
    "__repository__",
    "__version__",
]

__title__: str = "AlphaFold 3 input"

__package__: str = "alphafold3-input"
__module__: str = __package__

__version__: str = version(__package__)

__metadata__: PackageMetadata = metadata(__package__)
for item in __metadata__.json["project_url"]:
    match item:
        case _ if item.startswith("Repository"):
            __repository__: str = item.split(", ", 1)[1]
        case _ if item.startswith("Documentation"):
            __documentation__: str = item.split(", ", 1)[1]
        case _ if item.startswith("Issues"):
            __issues__: str = item.split(", ", 1)[1]
