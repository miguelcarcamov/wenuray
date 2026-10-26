from importlib.metadata import PackageNotFoundError, version

try:
    # Set by setuptools-scm from the git tags when the package is built or installed
    __version__ = version("wenuray")
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "0+unknown"
