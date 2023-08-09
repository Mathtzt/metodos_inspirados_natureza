import pandas as pd
import ProblemasOtimizacao as pot
import AlgoritmosOtimizacao as aot
import time
import matplotlib.pyplot as plt
from Utils import Utils as u

diretorio_saida = "./MaterialSaida/"
nome_experimento = "populacao_f8_150"

dimensao_problemas = 10
limite_inferior_problemas = [-100.0] * dimensao_problemas
limite_superior_problemas = [100.0] * dimensao_problemas

total_execucoes = 15

total_geracoes = 100
tamanho_populacao = 150
total_pais_cruzamento = 2
tipo_selecao_pais = "rws"
total_pais_manter_populacao = -1
total_pais_torneio = 3
tipo_cruzamento = "single_point"
taxa_cruzamento = 0.9
tipo_mutacao = "inversion"
taxa_mutacao = 0.05
elitismo = 2
salvar_melhores_solucoes = True
salvar_todas_solucoes = False

problemas = [
    # pot.HighConditionedEllipticFunction(dimensao=dimensao_problemas,
    #                                     limite_inferior=limite_inferior_problemas,
    #                                     limite_superior=limite_superior_problemas),
    pot.RastriginFunction(dimensao=dimensao_problemas,
                          limite_inferior=limite_inferior_problemas,
                          limite_superior=limite_superior_problemas)
]

dirpath = u.create_folder(path = diretorio_saida, name = nome_experimento, use_date = True)
imgs_path = u.create_folder(path = dirpath, name = 'imgs')

for problema in problemas:
    print("\n" + "-" * 80)
    print("> Problema: " + problema.nome)
    
    registro = u.create_opt_history(func_objetivo = problema.nome,
                                    total_geracoes = total_geracoes,
                                    tamanho_populacao = tamanho_populacao,
                                    total_pais_cruzamento = total_pais_cruzamento,
                                    tipo_selecao_pais = tipo_selecao_pais,
                                    tipo_cruzamento = tipo_cruzamento,
                                    tipo_mutacao = tipo_mutacao,
                                    taxa_mutacao = taxa_mutacao,
                                    elitismo = elitismo)

    for execucao in range(total_execucoes):
        print("\nExecução: " + str(execucao + 1))

        algoritmo_genetico = aot.AlgoritmoGenetico(problema=problema,
                                                   total_geracoes=total_geracoes,
                                                   tamanho_populacao=tamanho_populacao,
                                                   total_pais_cruzamento=total_pais_cruzamento,
                                                   tipo_selecao_pais=tipo_selecao_pais,
                                                   total_pais_manter_populacao=total_pais_manter_populacao,
                                                   total_pais_torneio=total_pais_torneio,
                                                   tipo_cruzamento=tipo_cruzamento,
                                                   taxa_cruzamento=taxa_cruzamento,
                                                   tipo_mutacao=tipo_mutacao,
                                                   taxa_mutacao=taxa_mutacao,
                                                   elitismo=elitismo,
                                                   salvar_melhores_solucoes=salvar_melhores_solucoes,
                                                   salvar_todas_solucoes=salvar_todas_solucoes,
                                                   semente_aleatoriedade=execucao)

        if algoritmo_genetico.valid_parameters:
            tempo_execucao = time.time()
            algoritmo_genetico.run()
            tempo_execucao = time.time() - tempo_execucao

            if algoritmo_genetico.run_completed:
                print("Geração Melhor Solução: " + str(algoritmo_genetico.best_solution_generation))
                print("Melhor Solução........: " + str(list(problema.ajusta_limites_variaveis(x=algoritmo_genetico.best_solutions[-1]))))
                print("Melhor Função Objetivo: " + str(1.0 / algoritmo_genetico.best_solutions_fitness[-1]))
                print("Melhor Avaliação......: " + str(algoritmo_genetico.best_solutions_fitness[-1]))
                print("Tempo Execução........: " + "%.2f" % tempo_execucao + " seg")
                
                if algoritmo_genetico.best_solutions_fitness[-1] < registro["melhor_avaliacao"]:
                    registro["melhor_avaliacao"] = algoritmo_genetico.best_solutions_fitness[-1]
                    registro["melhor_execucao"] = execucao

                plt.close("all")
                arquivo_grafico_evolucao = imgs_path + "/" + "Evolucao_" + "-".join(problema.nome.split()) + "_" + str(execucao + 1) + ".png"
                algoritmo_genetico.plot_fitness(title=problema.nome + " - Execução " + str(execucao + 1),
                                                xlabel="Geração",
                                                ylabel="Avaliação Melhor Solução",
                                                save_dir=arquivo_grafico_evolucao)

    df_registro = pd.DataFrame([registro])#.from_dict(registro, orient='columns')
    u.save_experiment_as_csv(base_dir = diretorio_saida, dataframe = df_registro, filename = 'opt_history')

    print(f"Melhor solução encontrada:\nExecução: {registro['melhor_execucao']}\nAvaliação: {registro['melhor_avaliacao']}")