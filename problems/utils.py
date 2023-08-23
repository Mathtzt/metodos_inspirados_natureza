class Utils:

    @staticmethod
    def read_matrix(path: str):
        with open(f'{path}.txt', 'r') as file:
            lines = file.readlines()

            # Inicializar uma lista vazia para armazenar a matriz
            matrix = []

            # Processar cada linha do arquivo
            for line in lines:
                # Dividir a linha nos espaços em branco e converter os valores em floats
                row = [float(value) for value in line.split()]
                matrix.append(row)

        return matrix
    
    @staticmethod
    def read_shift_data(path: str = './dados_auxiliares', name: str = 'shift_data_1'):
        loaded_list = []

        with open(f'{path}/{name}.txt', 'r') as fp:
            for line in fp:
                x = line[:-1]

                loaded_list.append(x)

        data_list = [float(value) for value in loaded_list[0].split()]

        return data_list