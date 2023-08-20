import os
import numpy as np
import pandas as pd
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
    
    @staticmethod
    def create_folder(path, name, use_date = False):
        """
        Método responsável por criar a pasta no diretório passado como parâmetro.
        """
        if use_date:
            dt = datetime.now()
            day = dt.strftime("%d")
            mes = dt.strftime("%m")
            hour = dt.strftime("%H")
            mm = dt.strftime("%M")
            dirname_base = f"{day}{mes}_{hour}{mm}_"
            directory = dirname_base + name
        else:
            directory = name

        parent_dir = path

        full_path = os.path.join(parent_dir, directory)

        if os.path.isdir(full_path):
            return full_path
        else:
            os.mkdir(full_path)
            return full_path

    @staticmethod
    def create_opt_history(execucao,
                           func_objetivo,
                           is_rotated,
                           dimensoes,
                           tamanho_populacao,
                           total_geracoes,
                           range_position,
                           omega,
                           reduce_omega_linearly,
                           range_speed,
                           cognitive_factor,
                           social_factor,
                           best_particle,
                           best_fitness,
                           out_bounds):
        d = {
            'execucao': execucao,
            'funcao_objetivo': func_objetivo,
            'is_rotated': is_rotated,
            'dimensoes': dimensoes,
            'tamanho_populacao': tamanho_populacao,
            'total_geracoes_realizadas': total_geracoes,
            'range_position': range_position,
            'omega': omega,
            'reduce_omega_linearly': reduce_omega_linearly,
            'range_speed': range_speed,
            'cognitive_factor': cognitive_factor,
            'social_factor': social_factor,
            'best_particle': best_particle,
            'best_fitness': best_fitness,
            'out_bounds': out_bounds
        }

        return d
    
    @staticmethod
    def save_experiment_as_csv(base_dir: str, dataframe: pd.DataFrame, filename: str):
        BASE_DIR = base_dir
        FILE_PATH = BASE_DIR + '/' + filename + '.csv'
        if not os.path.exists(FILE_PATH):
            dataframe.to_csv(FILE_PATH, index = False)
        else:
            df_loaded = pd.read_csv(FILE_PATH)
            dataframe_updated = pd.concat([df_loaded, dataframe], axis = 0)

            dataframe_updated.to_csv(FILE_PATH, index = False)

    @staticmethod
    def calc_mean_euclidian_distance_particles_from_pop(population):
        # Inicializando uma matriz para armazenar as distâncias
        num_vectors = len(population)
        distances = np.zeros((num_vectors, num_vectors))

        # Calcula as distâncias euclidianas entre os vetores
        for i in range(num_vectors):
            for j in range(i, num_vectors):
                distance = np.sqrt(np.sum((population[i] - population[j])**2))
                distances[i, j] = distance
                distances[j, i] = distance

        # Calcula a média das distâncias
        average_distance = np.mean(distances)

        return average_distance