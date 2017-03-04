__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/03/03 15:57:01"

""" Drawing samples from truncated 1-d normal distribution.

The truncated 1-d normal distribution can be restricated in interval (-inf, a],
[a, b] or [a, +inf]. The method used here is based on the paper "Simulation of 
truncated normal variables" by Christian P. Robert (doi:10.1007/BF00143942)
"""

import numpy as np

def _sample_right(a, mu = 0, sigma = 1.0):
    """Draw a smple from right open interval [a, +np.inf].

    Args:
        a: left boundary of the interval. It can be any number or -np.inf
        mu: mean value of the normal distribution.
        sigma: standard derivation of the normal distribution.

    Returns:
        A sample from the truncated normal distribution.

    Raises:
        ValueError: the error occurs when a = np.inf
    """    
    a = (a - mu) / sigma

    ## sampling
    if a <= 0:
        while True:
            x = np.random.normal()
            if x >= a: break
    elif a == -np.inf:
        return np.random.normal()
    elif a == np.inf:
        raise ValueError("The first parameter is invalid")
    else:
        alpha = 0.5 * (a + np.sqrt(a**2 + 4))
        while True:
            x = np.random.exponential(scale = alpha) + a
            rho = np.exp(-0.5*(x - a)**2)
            if np.random.uniform() <= rho:
                break
    return x * sigma + mu

def _sample_left(b, mu = 0, sigma = 1.0):
    """Draw a smple from left open interval (-np.inf, b).

    Args:
        b: right boundary of the interval. It can be any number or +np.inf
        mu: mean value of the normal distribution.
        sigma: standard derivation of the normal distribution.

    Returns:
        A sample from the truncated normal distribution.

    Raises:
        ValueError: the error occurs when b = -np.inf
    """    
    return -sample_right(-b, mu = mu, sigma = sigma)
    

def sample(a, b, mu = 0, sigma = 1.0):
    """Draw a smple from right open interval (a, b).

    Args:
        a: left boundary of the interval. It can be any number or -np.inf
        b: right boundary of the interval. It can be any number or +np.inf
        mu: mean value of the normal distribution.
        sigma: standard derivation of the normal distribution.

    Returns:
        A sample from the truncated normal distribution.

    Raises:
        ValueError: the error occurs when a = np.inf or b = -np.inf or a >= b
    """
    if a >= b:
        raise ValueError("The left boundary of truncated interval is larger \
        than the right boundary")
    if a == -np.inf:
        return _sample_left(b, mu = mu, sigma = sigma)
    elif b == np.inf:
        return _sample_right(a, mu = mu, sigma = sigma)
    elif a == np.inf:
        raise ValueError("The left boundary of the interval is Inf")
    elif b == -np.inf:
        raise ValueError("The right boudnary of the interval is -Inf")
    else:
        a = (a - mu) / sigma
        b = (b - mu) / sigma

        ## when the interval covers the mean value
        if a * b <= 0:
            if b - a >= 2:
                while True:
                    x = np.random.normal()
                    if x >= a and x <= b:
                        return x * sigma + mu
            else:
                while True:
                    x = np.random.uniform(a, b)
                    if np.random.uniform() <= np.exp(-0.5*x**2):
                        return x * sigma + mu
                    
        ## when the interval does not cover the mean value
        else:
            if a > 0 and b > 0:
                low = a
                up = b
            elif a < 0 and b < 0:
                low = np.abs(b)
                up = np.abs(a)

            cutoff = low + 2*np.sqrt(np.exp(1)) / (low + np.sqrt(low**2 + 4)) \
                     * np.exp((low**2 - low*np.sqrt(low**2+4))/4)

            ## when the interval is big enough
            if up > cutoff:
                while True:
                    x = _sample_right(low, mu = 0, sigma = 1)
                    if x <= up: break
            ## when the interval is not big enough
            else:
                while True:
                    x = np.random.uniform(low, up)
                    if np.random.uniform() < np.exp(-0.5*(x**2 - low**2)):
                        break

            if a > 0 and b > 0:
                return x * sigma + mu
            else:
                return -x * sigma + mu

