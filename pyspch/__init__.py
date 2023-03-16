__all__ = ["core","sp","display","nn","dtw"]
__magic__ = "a magic number"
from spch_version import __version__ 
#
# makes all core modules directly available
from .core import  audio, timit, hillenbrand, sequence_data
from .core.constants import *
from .core.file_tools import *
from .core.utils import *

# makes subpackages available at the top level
#from . import core
from . import sp
from . import display
#


