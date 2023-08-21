import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opfunu.cec.cec2014.function import F1, F8

from .utils import Utils as u

class DE:
    def __init__(self,
                 func_name: str = 'f1',
                 dimensions: int = 10,
                 population_size: int = 10,
                 max_evaluations: int = 50000,
                 bounds: list[int] = [-100, 100],
                 perc_mutation: float = .8,
                 perc_crossover: float = .7,
                 rotate_functions: bool = True,
                 early_stop_patience: int = None,
                 show_log: bool = True
                 ):
        self.func_name = func_name
        self.dimensions = dimensions
        self.population_size = population_size
        self.max_evaluations = max_evaluations // population_size
        self.bounds = bounds
        self.perc_mutation = perc_mutation
        self.perc_crossover = perc_crossover
        self.rotate_functions = rotate_functions
        self.early_stop_patience = early_stop_patience
        self.show_log = show_log

        self.nout_bounds = 0

    def get_func(self):
        selected_func = None

        if self.func_name == 'f1':
            if self.rotate_functions:
                selected_func = F1
            else:
                selected_func = u.f1_basic
        if self.func_name == 'f8':
            if self.rotate_functions:
                selected_func = F8
            else:
                selected_func = u.f8_basic
        
        if selected_func is None:
            raise Exception("Função selecionada não está implementada ainda.")

        return selected_func
    
    def mutation(self, X):
        x_base = X[0]
        x_r2 = X[1]
        x_r3 = X[2]

        return x_base + self.perc_mutation * (x_r2 - x_r3)
    
    def get_bounds_of_all_dim(self):
        return np.array([self.bounds] * self.dimensions)
    
    def check_bounds(self, mutated):
        nout_bounds = False
        for idx, val in enumerate(self.get_bounds_of_all_dim()):
            if mutated[idx] < val[0]:
                mutated[idx] = val[0]
                nout_bounds = True
                self.nout_bounds += 1
            if mutated[idx] > val[1]:
                mutated[idx] = val[1]
                nout_bounds = True
        
        if nout_bounds:
            self.nout_bounds += 1
            
        return mutated
    
    def crossover(self, mutated, target):
        p = np.random.rand(self.dimensions)

        vtrial = [mutated[i] if p[i] < self.perc_crossover else target[i] for i in range(self.dimensions)]
        
        return vtrial

    def init_population(self):
        all_bounds = self.get_bounds_of_all_dim()
        
        return all_bounds[:, 0] + (np.random.rand(self.population_size, len(all_bounds)) * (all_bounds[:, 1] - all_bounds[:, 0]))
    
    def main(self):
        # selecionando função desejada
        func = self.get_func()
        #
        population = self.init_population()
        #
        obj_initial = [func(ind) for ind in population]
        #
        best_vector = population[np.argmin(obj_initial)]
        best_obj = min(obj_initial)
        prev_obj = best_obj
        #
        for i in range(self.max_evaluations):
            #
            for j in range(self.population_size):
                #
                candidates = [candidate for candidate in range(self.population_size) if candidate != j]
                a, b, c = population[np.random.choice(candidates, 3, replace = False)]
                #
                mutated = self.mutation([a, b, c])
                #
                mutated = self.check_bounds(mutated)
                #
                vtrial = self.crossover(mutated, population[j])
                #
                obj_target = func(population[j])
                #
                obj_trial = func(vtrial)
                #
                if obj_trial < obj_target:
                    #
                    population[j] = vtrial
                    #
                    obj_initial[j] = obj_trial
        #
        best_obj = min(obj_initial)
        #
        if best_obj < prev_obj:
            best_vector = population[np.argmin(obj_initial)]
            prev_obj = best_obj
            #
            print('Iteration %d:  %s = %.5f' % (i, list(np.around(best_vector, decimals=5)), best_obj))
        
        return [best_vector, best_obj]