"""PhIP-seq classification and model-interpretation tools."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("phipml")
except PackageNotFoundError:
    # Useful when importing directly from an unpacked source tree.
    __version__ = "0+unknown"


__all__ = ["__version__"]