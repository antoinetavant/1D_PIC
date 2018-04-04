

import numpy as np
import scipy as sp
import astropy

from numba import jit
import random

from constantes import(me, q,kb,eps_0,mi)

@jit('f8(f8,f8)')
def generate_maxw(T, m):
    v_Te = np.sqrt(q*T/m)
    W = 2
    while (W >= 1 or W <= 0):
        R1 = (random.random()*2 -1 )
        R2 = (random.random()*2 -1 )

        W = R1**2 + R2**2
    W = np.sqrt( (-2*np.log(W))/W)

    v = v_Te*R1*W
    return v

def velocity_maxw_flux(T, m):
    import random
    v_Te = np.sqrt(q*T/m)

    R = random.random()
    v = (v_Te*np.sqrt(-np.log(R)))
    return v

def max_vect(N, T, m):

    return np.array([generate_maxw(T, m) for i in np.arange(N)])

def fux_vect(N, T, m):
    from numpy.random import rand
    v_Te = np.sqrt(q*T/m)

    return v_Te*np.sqrt(-np.log(rand(N)))

@jit('f8[:](i8,f8[:],f8[:],f8[:],f8)')
def numba_return_density(Np, partx, tabx, n, dx):

    for i in np.arange(Np):
        j = int(partx[i]/dx)
        deltax = tabx[j] - partx[i]
        n[j-1] += (1 - deltax)
        n[j  ] += (    deltax)

    return n
@jit('f8[:](i8,f8[:],f8[:],f8[:],f8[:],f8)')
def numba_return_meanv(Np, partx, partv, tabx, mean_v, dx):

    for i in np.arange(Np):
        j = int(partx[i]/dx)
        deltax = tabx[j] - partx[i]
        mean_v[j-1] += (1 - deltax)*partv[i]
        mean_v[j  ] += (    deltax)*partv[i]

    return mean_v

@jit('f8[:](i8,f8[:],f8[:],f8[:],f8[:],f8)')
def numba_return_stdv(Np, partx, partv, tabx, std_v, dx):

    for i in np.arange(Np):
        j = int(partx[i]/dx)
        deltax = tabx[j] - partx[i]
        std_v[j-1] += (1 - deltax)*partv[i]**2
        std_v[j  ] += (    deltax)*partv[i]**2

    return std_v

def smooth(x):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma=5)
