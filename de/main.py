import numpy as np
from classes.de import DE
from classes.utils import Utils

N_EXECUÇÕES = 10

root_path = Utils.create_folder(path = "./", name = "results")
dirpath = Utils.create_folder(path = root_path, name = "de", use_date = True)
imgs_path = Utils.create_folder(path = dirpath, name = 'imgs')

for exec in range(1, N_EXECUÇÕES + 1):
	print(f"######### Execução {exec} #########")
	de = DE(func_name = 'f1',
			max_evaluations = 100000,
			rotate_functions = False,
			early_stop_patience = 10,
			show_log = True)

	de.main(nexecucao = exec, dirpath = dirpath)
	de.plot_fitness_evolution(imgs_path = imgs_path, img_name = f"de_exec_{exec}")
	de.plot_ind_distance_evolution(imgs_path = imgs_path, img_name = f"de_distance_ind_{exec}")