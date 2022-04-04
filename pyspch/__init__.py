__all__ = ["core","sp","display","models","Densities","dtw","GaussianMixtureClf","libhmm"]
__magic__ = "a magic number"
from spch_version import __version__ 
import os
os.environ['__PYSPCH_PLT__'] = 'mpl'
#
# makes all core modules directly available
from .core import *
# makes subpackages available at the top level
from . import sp
from . import core
from . import display
from . import nn
#
#from .core.constants import *
#from .core.file_tools import *
#from .core.utils import *

