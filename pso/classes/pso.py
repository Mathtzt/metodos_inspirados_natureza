from deap import base, creator, tools

import random
import numpy as np
import matplotlib.pyplot as plt

from opfunu.cec.cec2014.function import F1, F8
from .utils import Utils as u

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)

class PSO:
    def __init__(self, 
                 func_name: str = 'f1',
                 dimensions: int = 10,
                 population_size: int = 10,
                 max_evaluations: int = 50000,
                 bounds: list[int] = [-100, 100],
                 omega: float = .9,
                 reduce_omega_linearly: bool = False,
                 min_speed: float = -0.5,
                 max_speed: float = 3.,
                 cognitive_update_factor: float = 2.,
                 social_update_factor: float = 2.,
                 rotate_functions: bool = True,
                 early_stop_patience: int = None,
                 show_log: bool = True
                 ):
        self.func_name = func_name
        self.dimensions = dimensions
        self.population_size = population_size
        self.max_evaluations = max_evaluations #// population_size
        self.min_start_position = bounds[0]
        self.max_start_position = bounds[1]
        self.omega = omega
        self.reduce_omega_linearly = reduce_omega_linearly
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.cognitive_update_factor = cognitive_update_factor
        self.social_update_factor = social_update_factor
        self.rotate_functions = rotate_functions
        self.early_stop_patience = early_stop_patience
        self.show_log = show_log

        self.nout_bounds = 0
        self.min_fitness_values = None
        self.avg_fitness_values = None
        self.toolbox = base.Toolbox()

    def define_as_minimization_problem(self):
        creator.create(name = "FitnessMin",
                       base = base.Fitness,
                       weights = (-1., ))
        
    def creating_particle_class(self):
        creator.create(name = 'Particle',
                       base = np.ndarray,
                       fitness = creator.FitnessMin,
                       speed = None,
                       best = None)

    def create_particle(self):
        particle = creator.Particle(np.random.uniform(low = self.min_start_position,
                                                      high = self.max_start_position,
                                                      size = self.dimensions))
        
        particle.speed = np.random.uniform(low = self.min_speed,
                                           high = self.max_speed,
                                           size = self.dimensions)
        
        return particle
    
    def creating_particle_register(self):
        self.toolbox.register(alias = 'particleCreator',
                              function = self.create_particle)
                        
        self.toolbox.register('populationCreator', tools.initRepeat, list, self.toolbox.particleCreator)

    def update_particle(self, particle, best, func_name):
        if func_name in ('f1', 'f8'):
            local_update_factor = self.cognitive_update_factor * np.random.uniform(0, 1, particle.size)
            global_update_factor = self.social_update_factor * np.random.uniform(0, 1, particle.size)
        else:
            raise Exception("Função não implementada ainda.")

        local_speed_update = local_update_factor * (particle.best - particle)
        global_speed_update = global_update_factor * (best - particle)

        particle.speed = (self.omega * particle.speed) + (local_speed_update + global_speed_update)
        # verificando se a nova posição sairá do espaço de busca. Se sim, ajustando para os limites.     
        out_bounds = False
        for i, speed in enumerate(particle.speed):
            if speed > self.max_speed:
                out_bounds = True
                particle.speed[i] = self.max_speed
            if speed < self.min_speed:
                out_bounds = True
                particle.speed[i] = self.min_speed
        
        if out_bounds:
            self.nout_bounds += 1

        # atualizando posição
        particle[:] = particle + particle.speed

    def register_to_update_particles(self):
        self.toolbox.register(alias = 'update',
                              function = self.update_particle,
                              func_name = self.func_name)

    def main(self, reset_classes: bool = False):
        if self.func_name == 'f1':
            if self.rotate_functions:
                self.toolbox.register(alias = 'evaluate', function = F1)
            else:
                self.toolbox.register(alias = 'evaluate', function = u.f1_basic)
        elif self.func_name == 'f8':
            if self.rotate_functions:
                self.toolbox.register(alias = 'evaluate', function = F8)
            else:
                self.toolbox.register(alias = 'evaluate', function = u.f8_basic)
        else:
            raise Exception("Função não implementada ainda.")
        
        ## inicializações
        self.define_as_minimization_problem()
        self.creating_particle_class()
        self.create_particle()
        self.creating_particle_register()
        self.register_to_update_particles()
        
        ## criando a população
        population = self.toolbox.populationCreator(n = self.population_size)
        ## criando objeto para salvar as estatísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'evals'] + stats.fields
        ## loop
        best = None
        initial_omega = self.omega
        nearly_stop_counter = 0
        igeneration_stopped = None
        for idx, generation in enumerate(range(1, self.max_evaluations + 1)):
            # reduzindo omega linearmente
            if self.reduce_omega_linearly:
                self.omega = initial_omega - (idx * (initial_omega - 0.4) / (self.max_evaluations))
            # avaliar todas as partículas na população
            for particle in population:
                # calcular o valor de fitness da partícula / avaliação
                particle.fitness.values = (self.toolbox.evaluate(particle), )
                # atualizando melhor partícula global
                if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                    particle.best = creator.Particle(particle)
                    particle.best.fitness.values = particle.fitness.values
                # atualizando valor global
                if best is None or best.size == 0 or best.fitness < particle.fitness:
                    best = creator.Particle(particle)
                    best.fitness.values = particle.fitness.values
        
                if np.min(particle.fitness.values) == 0.0:
                    nearly_stop_counter += 1
            # atualizando velocidade e posição
            for particle in population:
                self.toolbox.update(particle, best)
            # early stopping
            if self.early_stop_patience and nearly_stop_counter > self.early_stop_patience:
                igeneration_stopped = idx
                break
            # salvando as estatísticas
            if generation == 1 or generation % 500 == 0:
                logbook.record(gen = generation,
                                evals = len(population),
                                **stats.compile(population))
                if self.show_log:
                    if self.reduce_omega_linearly:
                        print(logbook.stream + f" | omega = {self.omega}")
                    else:
                        print(logbook.stream)

        self.min_fitness_values = [logbook[i]['min'] for i in range(len(logbook))]
        self.avg_fitness_values = [logbook[i]['avg'] for i in range(len(logbook))]

        print("-- Melhor partícula = ", best)
        print("-- Melhor fitness = ", best.fitness.values[0])
        print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)

        if reset_classes:
            del creator.FitnessMin
            del creator.Particle

    def plots(self, imgs_path: str, img_name: str):
        plt.figure()
        plt.plot(self.min_fitness_values, color = 'red')
        plt.plot(self.avg_fitness_values, color = 'green')
        plt.xlabel('Gerações')
        plt.ylabel('Min / Avg Fitness')
        plt.title('Fitness mínimo e médio através das gerações')
        plt.yscale('log')
        plt.legend(['Min', 'Avg'])
        # plt.tight_layout()
        filename = f'{imgs_path}/{img_name}.jpg' 
        # Salve a imagem
        plt.savefig(filename)