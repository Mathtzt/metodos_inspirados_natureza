import math

class BaseProblemaOtimizacao():
    def __init__(self,
                 nome,
                 dimensao,
                 limite_inferior,
                 limite_superior):
        self.nome = nome
        self.dimensao = dimensao
        self.limite_inferior = limite_inferior
        self.limite_superior = limite_superior

    def ajusta_limites_variaveis(self, x):
        aux = [x[i] for i in range(self.dimensao)]

        for i in range(self.dimensao):
            if aux[i] < self.limite_inferior[i]:
                aux[i] = self.limite_inferior[i]
            elif aux[i] > self.limite_superior[i]:
                aux[i] = self.limite_superior[i]

        return aux

    def funcao_objetivo(self, x):
        aux = self.ajusta_limites_variaveis(x=x)
        return None

class HighConditionedEllipticFunction(BaseProblemaOtimizacao):
    def __init__(self,
                 nome="High Conditioned Elliptic Function",
                 dimensao=5,
                 limite_inferior=[-100.0]*5,
                 limite_superior=[100.0]*5):
        BaseProblemaOtimizacao.__init__(self=self,
                                        nome=nome,
                                        dimensao=dimensao,
                                        limite_inferior=limite_inferior,
                                        limite_superior=limite_superior)

    def funcao_objetivo(self, x):
        aux = self.ajusta_limites_variaveis(x=x)

        func_obj = 0.0
        base_potencia = pow(10.0, 6.0)
        for i in range(self.dimensao):
            expoente_potencia = i / (self.dimensao - 1)
            func_obj += pow(base_potencia, expoente_potencia) * pow(aux[i], 2.0)

        return func_obj

class RastriginFunction(BaseProblemaOtimizacao):
    def __init__(self,
                 nome="Rastrigin Function",
                 dimensao=5,
                 limite_inferior=[-100.0]*5,
                 limite_superior=[100.0]*5):
        BaseProblemaOtimizacao.__init__(self=self,
                                        nome=nome,
                                        dimensao=dimensao,
                                        limite_inferior=limite_inferior,
                                        limite_superior=limite_superior)

    def funcao_objetivo(self, x):
        aux = self.ajusta_limites_variaveis(x=x)

        func_obj = 0.0
        for i in range(self.dimensao):
            func_obj += pow(aux[i], 2.0) - (10.0 * math.cos(2.0 * math.pi * aux[i])) + 10.0

        return func_obj