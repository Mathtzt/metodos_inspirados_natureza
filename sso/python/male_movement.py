import numpy as np

def ma_move(fn, mn, fsp, msp, femass, mamass, d, lb, ub, pm):
    # Preliminaries
    dt = np.zeros(mn)
    
    # Scale for distance
    scale = (-lb[0] + ub[0])  # / 2
    Indb = np.where(mamass >= np.median(mamass))[0]  # Male spiders above median
    new_msp = np.copy(msp)  # Create a copy to store the updated msp
    
    for i in range(mn):
        if i in Indb:  # Spider above the median
            # Start looking for a female with stronger vibration
            for j in range(fn):
                if femass[j] > mamass[i]:
                    # Calculate the distance
                    dt[j] = np.linalg.norm(msp[i, :] - fsp[j, :])
                else:
                    dt[j] = 0
            # Choose the shortest distance
            Ind = np.where(dt > 0)[0]
            val = dt[Ind]
            Imin = np.argmin(val)
            Ish = Ind[Imin]
            
            # Update moves
            if len(val) == 0:
                Vib = 0
                spaux = np.zeros(d)
            else:
                dt = dt / scale
                Vib = 2 * femass[Ish] * np.exp(-(np.random.rand() * dt[Ish] ** 2))
                spaux = fsp[Ish, :]
            
            delta = 2 * np.random.rand(d) - 0.5
            tmpf = 2 * pm * (np.random.rand(d) - 0.5)
            new_msp[i, :] = msp[i, :] + Vib * (spaux - msp[i, :]) * delta + tmpf
        else:
            # Spider below median, go to weighted mean
            spdpos = np.concatenate((fsp, msp), axis=0)
            spdwei = np.concatenate((femass, mamass))
            weigth = np.tile(spdwei, (d, 1)).T
            wmean = np.sum(weigth * spdpos, axis=0) / np.sum(weigth, axis=0)
            
            # Move
            delta = 2 * np.random.rand(d) - 0.5
            tmpf = 2 * pm * (np.random.rand(d) - 0.5)
            new_msp[i, :] = msp[i, :] + (wmean - msp[i, :]) * delta + tmpf
        
        # Check limits
        for j in range(mn):
            for k in range(d):
                new_msp[j, k] = max(lb[k], min(ub[k], new_msp[j, k]))
    
    return new_msp
