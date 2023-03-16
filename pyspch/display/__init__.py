__all__ = ["display","interactive"]

# import/use low-level API depending on backend
import os
try: os.environ['PYSPCH_BACKEND'] 
except: os.environ['PYSPCH_BACKEND'] = "mpl"

# setting certain plotting and printing preferences
import matplotlib as mpl
from cycler import cycler
# colors and markers for 15 classes 
markers = ('o', 'P','v', '^', '<', '>', 'X','8', 's', '*', 'h', 'H', 'D', 'd', 'p')
colors = ['blue','green','red','magenta','cyan','gold', 
          'darkorange','navy', 'teal','black',
          'maroon','lightblue','darkkhaki','grey','limegreen']

mpl.rcParams['axes.prop_cycle'] = cycler(color=colors) ## + cycler(marker=markers) )
mpl.rcParams['figure.figsize'] = [10.0, 10.0]
mpl.rcParams['font.size'] = 10
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'

# 
from .display import *
from .interactive import *