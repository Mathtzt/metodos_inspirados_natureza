import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

class Utils(ABC):
    @abstractmethod
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
        try: 
            os.mkdir(full_path)
            return full_path
        except: 
            return
        
    @abstractmethod
    def create_opt_history(func_objetivo,
                           total_geracoes,
                           tamanho_populacao,
                           total_pais_cruzamento,
                           tipo_selecao_pais,
                           tipo_cruzamento,
                           tipo_mutacao,
                           taxa_mutacao,
                           elitismo):
        d = {
            'funcao_objetivo': func_objetivo,
            'total_geracoes': total_geracoes,
            'tamanho_populacao': tamanho_populacao,
            'total_pais_cruzamento': total_pais_cruzamento,
            'tipo_selecao_pais': tipo_selecao_pais,
            'tipo_cruzamento': tipo_cruzamento,
            'tipo_mutacao': tipo_mutacao,
            'taxa_mutacao': taxa_mutacao,
            'elitismo': elitismo,
            'melhor_execucao': 0, 
            'melhor_avaliacao': 1e3
        }

        return d
    
    @abstractmethod
    def save_experiment_as_csv(base_dir: str, dataframe: pd.DataFrame, filename: str):
        BASE_DIR = base_dir
        FILE_PATH = BASE_DIR + filename + '.csv'
        if not os.path.exists(FILE_PATH):
            dataframe.to_csv(FILE_PATH, index = False)
        else:
            df_loaded = pd.read_csv(FILE_PATH)
            dataframe_updated = pd.concat([df_loaded, dataframe], axis = 0)

            dataframe_updated.to_csv(FILE_PATH, index = False)