import numpy as np
from utils import Utils

class CEC14:
    def __init__(self):
        """"""
        pass

    def __f1_base_func(self, particle):
        dim = len(particle)
        assert dim > 0, "A dimensão deve ser maior que zero."

        particle_flattened = np.array(particle).ravel()
        ndim = len(particle_flattened)
        idx = np.arange(0, ndim)

        return np.sum((10 ** 6) ** (idx / (ndim - 1)) * particle_flattened ** 2)

    def __f1_rotated(self, particle, f_matrix, f_shift):
        
        z = np.dot(f_matrix, particle - f_shift)

        return self.__f1_base_func(z) + 100,

    def f1(self, particle, ndim: int = 10):
        f_matrix = Utils.read_matrix("./M_1_D10")
        f_shift = Utils.read_shift_data()[:ndim]

        return self.__f1_rotated(particle = particle, f_matrix = f_matrix, f_shift = f_shift)
    
    def __f8_base_func(self, particle):
        dim = len(particle)
        assert dim > 0, "A dimensão deve maior que zero."

        x = np.array(particle).ravel()
        return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

    def __f8_rotated(self, particle, f_shift):
        
        z = 5.12 * (particle - f_shift) / 100

        return self.__f8_base_func(z) + 100, 

    def f8(self, particle, ndim: int = 10):
        f_shift = Utils.read_shift_data()[:ndim]

        return self.__f8_rotated(particle = particle, f_shift = f_shift)