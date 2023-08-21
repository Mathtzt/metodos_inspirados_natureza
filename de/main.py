import numpy as np
from classes.de import DE

de = DE(func_name = 'f1',
        rotate_functions = False,
        max_evaluations = 100)

solution = de.main()
print('\nSolution: %s = %.5f' % (list(np.around(solution[0], decimals=5)), solution[1]))