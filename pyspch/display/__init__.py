__all__ = ["display","interactive"]

# import low-level API depending on backend
import os
try: os.environ['PYSPCH_BACKEND'] 
except: os.environ['PYSPCH_BACKEND'] = "mpl"
#
#if os.environ['PYSPCH_BACKEND'] == "mpl":
#    from .display_mpl import *
#elif os.environ['PYSPCH_BACKEND']== "plotly":
#    print("pyspch(display): using PLOTLY backend !")
#    from .display_ly import *
    
from .display import *
from .interactive import *