def survive(fsp, msp, ofspr, fefit, mafit, spfit, fun, fn, dims):
    n1, _ = ofspr.shape
    
    # Evalute the offspring
    offit = [fun(ofspr[j, :], dims) for j in range(n1)]
    
    for i in range(n1):
        # Calculate the worst spider
        w1, w2 = max(spfit), spfit.index(max(spfit))
        
        # If the offspring is better than the worst spider
        if offit[i] < w1:
            # Check if is male or female
            if w2 >= fn:
                # Male
                msp[w2 - fn, :] = ofspr[i, :]
                mafit[w2 - fn] = offit[i]
            else:
                # Female
                fsp[w2, :] = ofspr[i, :]
                fefit[w2] = offit[i]
            
            spfit[w2] = offit[i]
