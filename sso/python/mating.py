import numpy as np

def mating(femass, mamass, fsp, msp, dims):
    # Generate the offsprings
    ofsp = np.empty((0, dims))
    cont = 1
    
    # Check whether a spider is good or not (above median)
    Indf = np.where(femass)[0]  # Female spiders
    Indm = np.where(mamass > np.median(mamass))[0]  # Male spiders above median
    
    fespid = fsp[Indf, :]  # Female spiders
    maspid = msp[Indm, :]  # Only the Male spiders above median
    sp2mate = []
    
    # Calculate the radio
    rad = np.zeros(dims)
    spid = np.concatenate((fsp, msp))
    for i in range(dims):
        rad[i] = np.max(spid[:, i]) - np.min(spid[:, i])
    r = (np.sum(rad) / 2) / dims
    
    # Start looking if there's a good female near
    sz = len(Indf)
    dist = np.zeros(sz)
    for i in range(len(Indm)):
        iaux = 1  # Aux to form the elements to mate
        for j in range(len(Indf)):
            dist[j] = np.linalg.norm(msp[Indm[i], :] - fsp[Indf[j], :])
        mate = []
        mass = []
        for k in range(len(Indf)):
            if dist[k] < r:
                mate.append(fsp[Indf[k], :])
                mass.append(femass[Indf[k]])
                iaux += 1
                # Form the matrix with elements to mate
                sp2mate = np.concatenate((msp[Indm[i], :], np.array(mate)), axis=0)
                masmate = np.concatenate(([mamass[Indm[i]]], mass))
        
        # Realizo el mate
        if sp2mate.size == 0:
            pass  # do nothing
        else:
            num2, n = sp2mate.shape
            for k in range(num2):
                for j in range(n):
                    accumulation = np.cumsum(masmate)
                    p = np.random.rand() * accumulation[-1]
                    chosen_index = np.where(accumulation > p)[0][0]
                    choice = chosen_index
                    # Form the new element
                    ofsp = np.vstack((ofsp, sp2mate[choice, :]))
                    cont += 1
    
    return ofsp
