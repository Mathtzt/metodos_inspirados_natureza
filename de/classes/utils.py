import numpy as np
import math

from datetime import datetime

class Utils:

    @staticmethod
    def f1_basic(particle):
        dim = len(particle)
        assert dim > 0, "A dimensão deve ser maior que zero."

        particle_flattened = np.array(particle).ravel()
        ndim = len(particle_flattened)
        idx = np.arange(0, ndim)

        return np.sum((10 ** 6) ** (idx / (ndim - 1)) * particle_flattened ** 2)    
        
    @staticmethod
    def f8_basic(particle):
        dim = len(particle)
        assert dim > 0, "A dimensão deve ser maior que zero."

        v = 0.0
        for i in range(dim):
            v += pow(particle[i], 2.0) - (10.0 * math.cos(2.0 * math.pi * particle[i])) + 10.0

        return v