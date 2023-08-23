import numpy as np
import matplotlib.pyplot as plt

def sso(spidn, itern):
    # Preliminares
    if spidn is None:
        spidn = 50
    if itern is None:
        itern = 500
    
    # Função Griewank
    def griewank(x):
        n = len(x)
        fr = 4000
        s = np.sum(x ** 2)
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
        fit = s / fr - p + 1
        return fit
    
    xd = -600
    xu = 600
    dims = 30
    lb = np.full((dims,), xd)
    ub = np.full((dims,), xu)
    
    # Parâmetros iniciais
    np.random.seed(0)  # Reset the random generator
    fpl = 0.65  # Lower Female Percent
    fpu = 0.9  # Upper Female Percent
    fp = fpl + (fpu - fpl) * np.random.rand()  # Aleatory Percent
    fn = round(spidn * fp)  # Number of females
    mn = spidn - fn  # Number of males
    
    pm = np.exp(np.linspace(0.1, 3, itern))  # Probabilities of attraction or repulsion
    
    fsp = np.random.uniform(lb, ub, (fn, dims))  # Initialize females
    msp = np.random.uniform(lb, ub, (mn, dims))  # Initialize males
    fefit = np.zeros(fn)  # Initialize fitness females
    mafit = np.zeros(mn)  # Initialize fitness males
    spwei = np.zeros(spidn)  # Initialize weight spiders
    
    # População de fêmeas e machos
    spfit = np.concatenate((fefit, mafit))  # Mix Females and Males
    bfitw = np.min(spfit)  # best fitness
    wfit = np.max(spfit)  # worst fitness
    
    # Calcular peso para cada aranha
    spwei = 0.001 + ((spfit - wfit) / (bfitw - wfit))
    fewei = spwei[:fn]  # Separar a massa das fêmeas
    mawei = spwei[fn:]  # Separar a massa dos machos
    
    # Inicializar memória do melhor
    Ibe = np.argmax(spwei)
    if Ibe >= fn:
        spbest = msp[Ibe - fn, :]
        bfit = mafit[Ibe - fn]
    else:
        spbest = fsp[Ibe, :]
        bfit = fefit[Ibe]
    
    spbesth = np.zeros((itern, dims))
    befit = np.zeros(itern)
    
    # Iterações
    for i in range(itern):
        # Movimento de aranhas fêmeas e machos
        fsp = fe_move(spidn, fn, fsp, msp, spbest, Ibe, spwei, dims, lb, ub, pm[i])
        msp = ma_move(fn, mn, fsp, msp, fewei, mawei, dims, lb, ub, pm[i])
        
        # Avaliação das funções para as fêmeas e machos
        fefit = np.array([griewank(x) for x in fsp])
        mafit = np.array([griewank(x) for x in msp])
        
        # Calcular pesos novamente
        spfit = np.concatenate((fefit, mafit))
        bfitw = np.min(spfit)
        wfit = np.max(spfit)
        spwei = 0.001 + ((spfit - wfit) / (bfitw - wfit))
        fewei = spwei[:fn]
        mawei = spwei[fn:]
        
        # Operador de acasalamento
        ofspr = mating(fewei, mawei, fsp, msp, dims)
        
        # Seleção de acasalamento
        if ofspr.size != 0:
            fsp, msp, fefit, mafit = survive(fsp, msp, ofspr, fefit, mafit, spfit, griewank, fn, dims)
            spfit = np.concatenate((fefit, mafit))
            bfitw = np.min(spfit)
            wfit = np.max(spfit)
            spwei = 0.001 + ((spfit - wfit) / (bfitw - wfit))
            fewei = spwei[:fn]
            mawei = spwei[fn:]
        
        # Memória do melhor global
        Ibe2 = np.argmax(spwei)
        if Ibe2 >= fn:
            spbest2 = msp[Ibe2 - fn, :]
            bfit2 = mafit[Ibe2 - fn]
        else:
            spbest2 = fsp[Ibe2, :]
            bfit2 = fefit[Ibe2]
        
        if bfit <= bfit2:
            bfit = bfit
            spbest = spbest
            befit[i] = bfit
        else:
            bfit = bfit2
            spbest = spbest2
            befit[i] = bfit
        
        spbesth[i, :] = spbest
        
        # Plot de resultados
        plt.plot(fsp[:, 0], fsp[:, 1], 'r.', msp[:, 0], msp[:, 1], 'bx', spbest[0], spbest[1], 'go')
        plt.axis([lb[0], ub[0], lb[1], ub[1]])
        plt.pause(0.01)
        plt.clf()
    
    # Plot dos resultados
    plt.figure()
    plt.plot(befit)
    plt.show()
