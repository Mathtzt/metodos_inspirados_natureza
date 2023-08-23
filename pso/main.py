from classes.pso import PSO
from classes.utils import Utils

N_EXECUÇÕES = 5

root_path = Utils.create_folder(path = "./", name = "results")
dirpath = Utils.create_folder(path = root_path, name = "pso", use_date = True)
imgs_path = Utils.create_folder(path = dirpath, name = 'imgs')

for exec in range(1, N_EXECUÇÕES + 1):
  print(f"######### Execução {exec} #########")
  pso = PSO(func_name = 'f1',
          max_evaluations = 50000,
          min_speed = -50.,
          max_speed = 50.,
          omega = .9,
          cognitive_update_factor = 2.,
          social_update_factor = 2.,
          reduce_omega_linearly = True,
          rotate_functions = False,
          early_stop_patience = 10,
          show_log = True
          )
  
  pso.main(reset_classes = True, nexecucao = exec, dirpath = dirpath)
  pso.plot_fitness_evolution(imgs_path = imgs_path, img_name = f"pso_exec_{exec}")
  pso.plot_particle_distance_evolution(imgs_path = imgs_path, img_name = f"pso_distance_particles_{exec}")