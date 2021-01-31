import numpy as np

def collapse(probas):
    """
    Parameters
    ----------
    probas: An array of probablities.
    
    Returns
    -------
    b: An array of binary states, whose dimensionality is identical to probas.
    """
    p = np.asarray(probas)
    b = p >= np.random.rand(*p.shape)
    return b.astype(int)

def logistic(x, temperature = 1.0):
    """
    Parameters
    ----------
    x: A numeric array.
    temperature: A non-negative number controlling the slope of the function.
    
    Returns
    -------
    y: The value of the function, which is often used as a probability.
    
    -------
    The function is numerically stable for very big/small values.
    """
    _x = np.asarray(x)
    if temperature == 0:
        # The logistic function is reduced to a step function.
        y = np.zeros(_x.shape)
        y[_x > 0] = 1.0
        y[_x == 0] = 0.5     
    else:
        norx = _x / temperature
        mask_p = norx >= 0
        mask_n = norx < 0        
        y = np.ones_like(norx)
        y[mask_p] = 1 / (1 + np.exp(-norx[mask_p]))
        # positive x gives small exp(-x): 1<denom<2
        z = np.zeros_like(y[mask_n])
        z = np.exp(norx[mask_n])
        y[mask_n] = z / (1 + z)        
        # negative x gives small exp(x)=z: 1<denom<2
    return y

def softmax(x, temperature = 1.0):
    """
    Parameters
    ----------
    x: A two-dimensional numeric array.
    temperature: A non-negative number controlling the slope of the function.
    
    Returns
    -------
    y: The value of the function, which is often used as a probability. Each row adds up to 1.
    
    -------
    The function is numerically stable for very big/small values.
    """
    _x = np.asarray(x)
    _p = []
    if temperature == 0:
        for _xrow in _x:
            _prow = np.zeros_like(_xrow)
            maxidx = np.nonzero(_xrow)
            _prow[maxidx] = 1 / maxidx[0].size
            _p.append(_prow)
    p = np.reshape(_p, _x.shape)
    return p
        
    #     for item in _x:
    #         prob = np.zeros(item_size)
    #         maxids 
    #         prob[maxids] = 1 / len(maxids)
    #         probs.append(prob)
    # else:
    #     for item in canx:
    #         noritem = (item - np.amax(item)) / temperature
    #         pitem = np.exp(noritem)
    #         prob = pitem / np.sum(pitem)
    #         probs.append(prob)            
             
    # y = canonicalisation(probs, item_size)
    # return y

def entropy(P, base = None):
    """
    Parameters
    ----------
    P: A discrete probability distribution listed in a numeric array.
    base: The logarithmic base when calculating entropy with the default value being e.
    
    Returns
    -------
    H: The entropy of the distribution P.
    """
    ps = P / np.sum(P)
    ps[ps == 0] = 1 # plog(p) = when p = 0 or 1
    if base is None:
        denom = 1
    else:
        denom = np.log(base)
    logps = np.log(ps) / denom
    return -np.dot(ps, logps)
    