import numpy as np

def fe_move(spidn, fn, fsp, msp, spbest, Ibe, spmass, d, lb, ub, pm):
    # Preliminaries
    dt1 = np.zeros(fn)
    dt2 = np.zeros(spidn - fn)
    
    # Scale for distance
    scale = (-lb[0] + ub[0])
    new_fsp = np.copy(fsp)  # Create a copy to store the updated fsp
    
    for i in range(fn):  # Move the females
        for j in range(fn):  # Check all the spiders
            if spmass[j] > spmass[i]:  # If there's someone more attractive
                # Calculate the distance
                dt1[j] = np.linalg.norm(fsp[i, :] - fsp[j, :])
            else:
                dt1[j] = 0
                
        for j in range(spidn - fn):
            if spmass[fn + j] > spmass[i]:
                # Calculate the distance
                dt2[j] = np.linalg.norm(fsp[i, :] - msp[j, :])
            else:
                dt2[j] = 0
        
        dt = np.concatenate((dt1, dt2)) / scale
        
        # Choose the shortest distance
        Ind = np.where(dt > 0)[0]
        val = dt[Ind]
        Imin = np.argmin(val)
        Ish = Ind[Imin]
        
        if Ish >= fn:
            # Is Male
            spaux = msp[Ish - fn, :]
        else:
            # Is Female
            spaux = fsp[Ish, :]
        
        if len(val) == 0:
            Vibs = 0
            spaux = np.zeros(d)
        else:
            Vibs = 2 * (spmass[Ish] * np.exp(-((np.random.rand() * dt[Ish]) ** 2)))
        
        if Ibe >= fn:
            # Is Male
            dt2 = np.linalg.norm(fsp[i, :] - msp[Ibe - fn, :])
        else:
            # Is Female
            dt2 = np.linalg.norm(fsp[i, :] - fsp[Ibe, :])
        
        dtb = dt2 / scale
        Vibb = 2 * (spmass[Ibe] * np.exp(-(np.random.rand() * dtb ** 2)))
        
        if np.random.rand() >= pm:
            # Do an attraction
            betha = np.random.rand(d)
            gamma = np.random.rand(d)
            tmpf = 2 * pm * (np.random.rand(d) - 0.5)
            new_fsp[i, :] = fsp[i, :] + (Vibs * (spaux - fsp[i, :]) * betha) + (Vibb * (spbest - fsp[i, :]) * gamma) + tmpf
        else:
            # Do a repulsion
            betha = np.random.rand(d)
            gamma = np.random.rand(d)
            tmpf = 2 * pm * (np.random.rand(d) - 0.5)
            new_fsp[i, :] = fsp[i, :] - (Vibs * (spaux - fsp[i, :]) * betha) - (Vibb * (spbest - fsp[i, :]) * gamma) + tmpf
    
    # Check limits
    for j in range(fn):
        for k in range(d):
            new_fsp[j, k] = max(lb[k], min(ub[k], new_fsp[j, k]))
    
    return new_fsp
