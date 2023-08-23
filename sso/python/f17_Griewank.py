import numpy as np

def f17_griewank(x):
    # Griewank Function
    # Domain: [-600, 600]
    # The global minima: x* = (0, ..., 0), f(x*) = 0.
    n = len(x)
    fr = 4000
    s = 0
    p = 1
    
    for i in range(n):
        s += x[i] ** 2
        
    for j in range(n):
        p *= np.cos(x[j] / np.sqrt(j + 1))  # j + 1 because Python indexing starts from 0
    
    fit = s / fr - p + 1
    return fit
