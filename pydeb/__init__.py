from importlib.metadata import version, PackageNotFoundError
from .model import *
from . import infer
from . import calibrate

try:
    __version__ = version("pydeb")
except PackageNotFoundError:
    pass
