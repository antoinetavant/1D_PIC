# a test file for the 1D_PIC code

import pytest
import numpy as np

me = 9.109e-31; #[kg] electron mass
q = 1.6021765650e-19; #[C] electron charge
kb = 1.3806488e-23;  #Blozman constant
eps_0 = 8.8548782e-12; #Vaccum permitivitty
mi = 131*1.6726219e27 #[kg]


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



def test_plasma():

    from plasma import plasma as pl

    do = pl(dT,Nx,Lx,1,n,Te,Ti)

    do.ele.x = np.array([0],dtype='float64')
    do.ele.V = np.array([[0,0,0]],dtype='float64')
    do.ion.x = np.array([0],dtype='float64')
    do.ion.V = np.array([[0,0,0]],dtype='float64')
    do.E[:,:] = 0
    do.pusher()

    assert do.ele.x == np.array([0],dtype='float64')
