""" Define an internal EPS for precision and floor protection against log's of 0 
    
    With flt32 arrays, a safe setting is EPS_FLOAT = 1.e-39 which is just within flt32 precision
        thus    EPS_LOG = -89.801 .  EPS_LOG10 = -39.000
        
    When inputs are flt64, it would be safe to set EPS to 1.e-300, and EPS_LOG = -690.7755
    
    
    For sampled data, flooring on the level of 16 bit signed integer makes more sense, for those
    situations one can use SIGEPS_FLOAT = 1.19e-7 and a corresponding dB value of SIGEPS_DB of -69.2
    """
#
import numpy as np
LOG10        = np.log(10)          # 2.302585092994046 
LOG2DB       = 10.0/LOG10

# flooring to apply in general computations
EPS_FLOAT    = 1.e-39
EPS_LOG      = -89.801             # np.log(EPS_FLOAT)
EPS_LOG10    = -39.0              # np.log10(EPS_FLOAT)

# flooring that can be applied to signals and dB levels ( from KALDI )
SIGEPS_DB    = -69.23689          # scale*math.log(1.19209290e-7)  default flooring applied in KALDI
SIGEPS_FLOAT = 1.19209290e-7
