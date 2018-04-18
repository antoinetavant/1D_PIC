# a test file for the 1D_PIC code

import pytest
import numpy as np

from constantes import(me, q,kb,eps_0,mi)

#parameters
Lx = 1 #System length
dX = 1e-5 #dX in m
Nx = int(Lx/dX)+1 #cell number
Lx = Nx*dX
print("Nx = {Nx}, and Lx = {Lx} cm".format(Nx = Nx, Lx = Lx*100))

n = 1e17  #[m^-3]
dT = 1e-12 #time step
Te = 20;     #[eV] Electron distribution temperature
Ti = 5 #[eV]

L_De = np.sqrt(eps_0*Te/(q*n))

dT = 1e-10 #time step


from plasma import plasma as pl
@pytest.mark.parametrize("Np, partsout", [
    (1,0),(10,0),(100,0),(100,10),(100,50)])
def test_pusher(Np,partsout):
    """Test the pusher with the particle DS (coumpt_out )
    """

    dT = 1
    do = pl(dT,Nx,Lx,Np,n,Te,Ti)
    do.ele.compt_out = partsout
    do.ion.compt_out = partsout

    do.ele.x = np.zeros(Np,dtype='float64')
    do.ele.V = np.ones((Np,3),dtype='float64')
    do.ion.x = np.zeros(Np,dtype='float64')
    do.ion.V = np.ones((Np,3),dtype='float64')
    do.E[:,:] = 0
    do.pusher()
    espected = np.ones(Np,dtype='float64')

    if partsout > 0:
        espected[-partsout:] = 0.0

    assert np.allclose(do.ele.x,espected)
    assert np.allclose(do.ion.x,espected)

@pytest.mark.parametrize("Nx,expected", [
    (1, 2),
    (20, 21),
    (500, 501),
])
def test_size(Nx,expected):

    do = pl(1.,Nx,1.,1,1.,1.,1.)

    assert len(do.phi) == expected
    assert len(do.x_j) == expected
    assert len(do.rho) == expected
    assert len(do.ni) == expected
    assert len(do.ne) == expected
    assert do.E.shape[0] == expected

@pytest.mark.parametrize("ne,Te, Nx, n_average,v", [
    (3.2e15, 50,10,7,5),
    (7e12, 21,100,13,1),
    (7e12, 21,100,13,13),
    (3.2e15, 50,10,7,51),
])
def test_diags(ne,Te,Nx,n_average,v):
    """generate a plasma of density ne and temperature Te,
    and test the diagnostics results"""

    Np = 100*Nx
    Lx = 0.37
    do = pl(0,Nx,Lx,Np,ne,Te,1,n_average = n_average)
    do.compute_rho()
    do.ele.V[:,0] = v
    for i in range(n_average):
        do.diags(i)

    measured_ne = do.data[do.lastkey]["ne"].mean()
    measured_Te = do.data[do.lastkey]["Te"].mean()
    measured_ve = do.data[do.lastkey]["ve"].mean()
    espected_ve = v
    espected_Te = 0 #v**2*me/(2*q)
    espected_ne = ne
    print(measured_Te, espected_Te,"   zjazofhzepfu   ")

    #print(do.data[do.lastkey]["ne"].mean(),ne)
    assert do.ele.Npart == Np
    assert do.ele.compt_out == 0
    assert do.lastkey == str(n_average-1)
    assert np.isclose(measured_ne,espected_ne,rtol=1e-2)
    assert np.isclose(measured_ve,espected_ve,rtol=1e-3)
    assert np.isclose(measured_Te,espected_Te,rtol=1e-3)

@pytest.mark.parametrize("ne,Te, Nx, n_average,", [
    (3.2e15, 50,530,7),
    (7e12, 21,100,13),
])
def test_Te(ne,Te,Nx,n_average):
    """generate a plasma of density ne and temperature Te,
    and test the diagnostics results"""
    Np = 100*Nx
    Lx = 0.37
    do = pl(0,Nx,Lx,Np,ne,Te,1,n_average = n_average)
    do.compute_rho()
    do.diags(0)

    espected_Te = Te
    measured_Te = do.Te.mean()

    assert np.isclose(measured_Te,espected_Te,rtol=5e-2)
