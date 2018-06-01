

import numpy as np

from numba import jit

import random

from constantes import q

numbaparallel = False


@jit('f8(f8,f8)')
def generate_maxw(T, m):

    v_Te = np.sqrt(q*T/m)
    W = 2
    while (W >= 1 or W <= 0):
        R1 = (random.random()*2 - 1)
        R2 = (random.random()*2 - 1)

        W = R1**2 + R2**2
    W = np.sqrt((-2*np.log(W))/W)

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


@jit('f8[:](f8[:])')
def numba_ones_like(x):
    return np.ones_like(x)


@jit('f8[:](f8[:],i8)')
def numba_power(x, p):
    return x**p


@jit('f8[:](i8,f8[:],f8[:],f8[:],f8[:],f8,i8)',
     nopython=True, parallel=numbaparallel)
def numba_return_part_diag(Np, partx, partv, tabx, diag, dx, power):
    """general function for the particle to grid diagnostics"""

    if power == 0:
        info = numba_ones_like(partv)
    elif power > 0:
        info = numba_power(partv, power)
    else:
        print("Unknow dignostics !!")
        return diag

    Jmax = len(tabx) - 1
    for i in range(Np):
        j = int(partx[i]/dx)
        if j > Jmax:
            j = Jmax
        if i >= 0 and j >= 0:
            deltax = abs(tabx[j] - partx[i])
            diag[j-1] += (1 - deltax)*info[i]
            diag[j] += deltax * info[i]

        else:
            print("i = ", i, ' over Np = ', Np, " and partx[i] = ", partx[i])
    return diag


@jit('f8[:](i8,f8[:],f8[:],f8[:],f8)')
def numba_return_density(Np, partx, tabx, n, dx):
    """wrapper for presperity"""
    return numba_return_part_diag(Np, partx, partx, tabx, n, dx, power=0)


# @jit('f8[:](i8,f8[:],f8[:],f8[:],f8[:],f8)')
# def numba_return_meanv(Np, partx, partv, tabx, mean_v, dx):
#     """wrapper for presperity"""
#     return numba_return_part_diag(Np, partx, partv, tabx, n, dx, power = 1)


# @jit('f8[:](i8,f8[:],f8[:],f8[:],f8[:],f8)')
# def numba_return_stdv(Np, partx, partv, tabx, std_v, dx):
#     """wrapper for presperity"""
#     return numba_return_part_diag(Np, partx, partv, tabx, n, dx, power=2)

@jit('i8[:](f8[:],f8)', nopython=True)
def normDx(tabx, dx):
    """devise a vector and use the `int` function to get integers
    """
    tmp = tabx/dx
    normedx = np.zeros(tabx.size, dtype=np.int64)
    for i in range(tmp.size):
        normedx[i] = int(tmp[i])
    return normedx


@jit('f8[:](f8[:],f8[:],f8[:], f8)', nopython=True, parallel=numbaparallel)
def numba_interp1D(partx, tabx, tabE, dx):
    """Compute the lineare interpolation of the electric field in
     the X directions
    This numba function should be faster than the scipy.interp1d and
     numpy.interp

    """
    partE = np.zeros_like(partx)
    normedx = np.zeros(tabx.size, dtype=np.int8)
    normedx = normDx(tabx, dx)

    maxX = int(normedx[-1])
    for i in np.arange(len(partx)):
        x = partx[i]/dx
        if x < 0:
            j = 0
            deltax = x
            partE[i] = (1-deltax)*tabE[0] + deltax*tabE[1]
        elif x > maxX:
            j = maxX
            deltax = (x - maxX)
            partE[i] = (1+deltax)*tabE[j] - deltax*tabE[j-1]
        else:
            j = int(x)  # position of the particle, in intex of tabx
            deltax = abs(x - normedx[j])  # length to cell center
            partE[i] = (1-deltax)*tabE[j] + deltax*tabE[j+1]

    return partE


@jit('f8[:](f8[:],i8[:],f8[:])', nopython=True, parallel=numbaparallel)
def numba_interp1D_normed(partx, normedx, tabE):
    """Compute the lineare interpolation of the electric field in
     the X directions but with normed position
    This numba function should be faster than the
     scipy.interp1d and numpy.interp

    """
    partE = np.zeros_like(partx)
    maxX = int(normedx[-2])

    for i in np.arange(len(partx)):
        x = partx[i]
        if x < 0:
            deltax = x
            partE[i] = (1-deltax)*tabE[0] + deltax * tabE[1]
        elif x >= maxX:
            j = maxX
            deltax = (x - maxX)
            partE[i] = (1+deltax)*tabE[j] - deltax * tabE[j-1]
        else:
            j = int(x)  # position of the particle, in intex of tabx
            deltax = abs(x - normedx[j])  # length to cell center

            partE[i] = (1-deltax)*tabE[j] + deltax*tabE[j+1]

    return partE


def smooth(x):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma=5)


@jit('f8[:](f8[:],f8[:],f8[:],f8[:],i8)',
     nopython=True, parallel=numbaparallel)
def numba_thomas_solver(di, ai, bi, ciprim, Nx):
    """Solve thomas with the upward and download loops
    """

    diprim = di
    diprim[0] /= bi[0]

    for i in np.arange(1, len(diprim)):
        diprim[i] -= ai[i]*diprim[i-1]
        diprim[i] /= bi[i] - ai[i]*ciprim[i-1]

    # Init solution

    phi = np.zeros(Nx)
    phi[-1] = diprim[-1]
    # limit conditions
    # SOLVE
    for i in np.arange(Nx-2, -1, -1):
        phi[i] = diprim[i] - ciprim[i]*phi[i+1]

    return phi


@jit("i8(f8[:],f8[:,:],f8)",
     nopython=True, parallel=numbaparallel)
def popout(x, V, val):
    """move elements that do not correspond to the condition
    x > val to the end of the table.

    Inputs :
    =========
    x, v (In and out) : array of float64

    val: float64 the threshold

    return:
    =======
    compt: int64 number of elements put at the end of x

    """

    # init the parameters
    compt = 0
    N = len(x)-1
    zeros = np.zeros(3)
    # linear search from the end of the table
    for i in np.arange(N, 0, -1):
        pos = x[i]
        if V[i, 1] == 0.0:
            compt += 1
        else:
            if (pos >= val) or (pos <= 0.0):  # Condition to move the
                                            # element at the end
                # exchange the current element with the last
                tmp = x[N - compt - 1]
                x[N - compt - 1] = 0.0
                x[i] = tmp

                V[i, :] = V[N - compt - 1, :]
                V[N - compt - 1, :] = zeros

                compt += 1

    return compt


@jit("i8(f8[:],f8[:,:],f8)",
     nopython=True, parallel=numbaparallel)
def mirror_vect(x, V, val):
    """move elements that do not correspond to the condition
    x > val to the end of the table.

    Inputs :
    =========
    x, v (In and out) : array of float64

    val: float64 the threshold

    return:
    =======
    0 if sucess

    """

    # init the parameters
    compt = 0
    N = len(x)-1
    # linear search from the end of the table
    for i in np.arange(N, 0, -1):
        pos = x[i]
        if V[i, 0] == 0.0:
            compt += 1
        else:
            if (pos >= val) or (pos <= 0.0):    # Condition to change
                                                # the velocity oriantation
                V[i, 0] *= -1.0

                if pos <= 0.0:
                    x[i] = 0.0
                if pos >= val:
                    x[i] = val

    return 0
