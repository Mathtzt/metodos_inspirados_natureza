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
                 perc_mutation: float = .9,
                 perc_crossover: float = .4,
                 crossover_type: str = None,
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
        self.crossover_type = crossover_type
        self.rotate_functions = rotate_functions
        self.early_stop_patience = early_stop_patience
        self.show_log = show_log

        self.nout_bounds = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.mean_euclidian_distance_particles = []

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
        # gera vetor de limites por dimensão do problema
        return np.array([self.bounds] * self.dimensions)
    
    def check_bounds(self, mutated):
        # verificando se vetor de mutação passou dos limites do espaço de busca. Se sim, altera valor para o limite mais próximo.
        nout_bounds = False
        for idx, val in enumerate(self.get_bounds_of_all_dim()):
            if mutated[idx] < val[0]:
                mutated[idx] = val[0]
                nout_bounds = True
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
        # buscando vetor de limites x dim
        all_bounds = self.get_bounds_of_all_dim()
        # inicializando a população de soluções candidatas de forma aleatória dentro dos limites especificados.
        return all_bounds[:, 0] + (np.random.rand(self.population_size, len(all_bounds)) * (all_bounds[:, 1] - all_bounds[:, 0]))
    
    def main(self, nexecucao: int = 0, dirpath: str = './'):
        # selecionando função desejada
        func = self.get_func()
        # inicilizando população
        population = self.init_population()
        # avaliando a população inicial de soluções candidatas
        fitness = [func(ind) for ind in population]
        # guardando informações do melhor vetor
        best_vector = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        prev_fitness = best_fitness
        # informações para registro
        nearly_stop_counter = 0
        igeneration_stopped = None
        # inicializando loop das gerações
        for i in range(1, self.max_evaluations + 1):
            # iteração de todas as soluções candidatas da população
            for j in range(self.population_size):
                # escolhendo três candidatos (a, b e c), que não sejam o atual
                candidates = [candidate for candidate in range(self.population_size) if candidate != j]
                a, b, c = population[np.random.choice(candidates, 3, replace = False)]
                # realizando processo de mutação
                mutated = self.mutation([a, b, c])
                # verificando se vetor que sofreu a mutação saiu do espaço de busca. Se sim, aplica correção
                mutated = self.check_bounds(mutated)
                # realizando crossover
                vtrial = self.crossover(mutated, population[j])
                # calculando o valor da função objetivo para o vetor alvo
                fitness_target = func(population[j])
                # calculando o valor da função objetivo para o vetor candidato escolhido
                fitness_trial = func(vtrial)
                # realizando seleção
                if fitness_trial < fitness_target:
                    # substituindo o individuo da população pelo novo vetor
                    population[j] = vtrial
                    # armazenando o novo valor da função objetivo
                    fitness[j] = fitness_trial
            # encontrando o vetor com melhor desempenho em cada iteração
            best_fitness = min(fitness)
            avg_fitness = np.mean(fitness)
            std_fitness = np.std(fitness)
            if best_fitness == 0.0:
                nearly_stop_counter += 1
            # armazenando o valor mais baixo da função objetivo (problema de minimização)
            if best_fitness < prev_fitness:
                best_vector = population[np.argmin(fitness)]
                prev_fitness = best_fitness

            # printando progresso
            if self.show_log and i % 500 == 0:
                print(f'Gen {i} | Min {best_fitness} | Avg {std_fitness}')
                #%d:  %s = %f' % (i, list(np.around(best_vector, decimals=5)), best_fitness))

            # guardando informações para registro da otimização
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.mean_euclidian_distance_particles.append(u.calc_mean_euclidian_distance_particles_from_pop(population))
            # early stopping
            if self.early_stop_patience and nearly_stop_counter > self.early_stop_patience:
                igeneration_stopped = i
                break
            else:
                igeneration_stopped = i

        print("-- Melhor individuo = ", best_vector)
        print("-- Melhor fitness = ", best_fitness)
        print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)

        registro = u.create_opt_history(execucao = nexecucao,
                             func_objetivo = self.func_name,
                             is_rotated = self.rotate_functions,
                             dimensoes = self.dimensions,
                             tamanho_populacao = self.population_size,
                             total_geracoes = igeneration_stopped,
                             range_position = self.bounds,
                             perc_mutation = self.perc_mutation,
                             perc_crossover = self.perc_crossover,
                             crossover_type = self.crossover_type,
                             best_ind = best_vector,
                             best_fitness = best_fitness,
                             out_bounds = self.nout_bounds)
        
        df_registro = pd.DataFrame([registro])
        u.save_experiment_as_csv(base_dir = dirpath, dataframe = df_registro, filename = 'opt_history')

    def plot_fitness_evolution(self, imgs_path: str, img_name: str):
        plt.figure()
        plt.plot(self.best_fitness_history, color = 'red')
        plt.plot(self.avg_fitness_history, color = 'green')
        plt.xlabel('Gerações')
        plt.ylabel('Min / Avg Fitness')
        plt.title('Fitness mínimo e médio através das gerações')
        plt.yscale('log')
        plt.legend(['Min', 'Avg'])
        
        # Salve a imagem
        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)

    def plot_ind_distance_evolution(self, imgs_path: str, img_name: str):
        plt.figure()
        plt.plot(self.mean_euclidian_distance_particles, color = 'blue')
        plt.xlabel('Gerações')
        plt.ylabel('Avg')
        plt.title('Distância média das partículas')
        plt.yscale('log')
        plt.legend(['Avg'])

        # Salve a imagem
        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)