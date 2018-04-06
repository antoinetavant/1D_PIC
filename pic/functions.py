

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
    """We could improve this with a parameter "power of v" in order to refactor the code
       between density, meanV and std_v
       """

    for i in np.arange(Np):
        j = int(partx[i]/dx)
        deltax = tabx[j] - partx[i]
        std_v[j-1] += (1 - deltax)*partv[i]**2
        std_v[j  ] += (    deltax)*partv[i]**2

    return std_v

@jit('f8[:](f8[:],f8[:],f8[:], f8)')
def numba_interp1D(partx, tabx, tabE, dx):
    """Compute the lineare interpolation of the electric field in the X directions
    This numba function should be faster than the scipy.interp1d and numpy.interp

    """
    partE = np.zeros_like(partx)
    normedx = (tabx/dx).astype(int)
    maxX = int(normedx[-1])
    for i in np.arange(len(partx)):
        x = partx[i]/dx
        if x < 0:
            j = 0
            deltax = x
            partE[i] = ((1-deltax)*tabE[0] + ( deltax)*tabE[1])
        elif x > maxX:
            j = maxX
            deltax = (x - maxX)
            partE[i] = ((1+deltax)*tabE[j] - ( deltax)*tabE[j-1])
        else:
            j = int(x) #position of the particle, in intex of tabx
            deltax = abs(x - normedx[j]) #length to cell center
            partE[i] = ((1-deltax)*tabE[j] + ( deltax)*tabE[j+1])

    return partE


@jit('f8[:](f8[:],i8[:],f8[:])')
def numba_interp1D_normed(partx, normedx, tabE):
    """Compute the lineare interpolation of the electric field in the X directions but with normed position
    This numba function should be faster than the scipy.interp1d and numpy.interp

    """
    partE = np.zeros_like(partx)
    maxX = int(normedx[-1])

    for i in np.arange(len(partx)):
        x = partx[i]
        if x < 0:
            deltax = x
            partE[i] = ((1-deltax)*tabE[0] + ( deltax)*tabE[1])
        elif x > maxX:
            j = maxX
            deltax = (x - maxX)
            partE[i] = ((1+deltax)*tabE[j] - ( deltax)*tabE[j-1])
        else:
            j = int(x) #position of the particle, in intex of tabx
            deltax = abs(x - normedx[j]) #length to cell center
            partE[i] = ((1-deltax)*tabE[j] + ( deltax)*tabE[j+1])

    return partE

def smooth(x):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma=5)

@jit('f8[:](f8[:],f8[:],f8[:],f8[:],i8)')
def numba_thomas_solver(di,ai,bi,ciprim,Nx):
    """Solve thomas with the upward and download loops

    """

    diprim = di
    diprim[0] /= bi[0]

    for i in np.arange(1,len(diprim)):
        diprim[i] -= ai[i]*diprim[i-1]
        diprim[i] /= bi[i] - ai[i]*ciprim[i-1]

    #Init solution

    phi = np.zeros(Nx + 1)
    phi[-1] = diprim[-1]
    #limit conditions

    #SOLVE
    for i in np.arange(Nx-1,-1,-1):
        phi[i] = diprim[i] - ciprim[i]*phi[i+1]

    return phi
