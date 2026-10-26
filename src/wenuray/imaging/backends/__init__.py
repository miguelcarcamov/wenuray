"""
Built-in imagers that follow the contract of wenuray.imaging.contract (#23). Optional backends, such as
Pyralysis (#24), are imported lazily so their dependencies stay optional.
"""
from .pyralysis import Pyralysis
from .tclean import Tclean
