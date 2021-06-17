"""
Different learning rates
"""

import math 

def constant(eta):
    x = lambda t : eta 
    return x

def diminishing(gamma, alpha):
    return lambda t : gamma / (1 + t)**(alpha)