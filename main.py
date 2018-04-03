#import
import numpy as np
import scipy as sp
import astropy

import matplotlib.pyplot as plt

import pic

from pic.plasma import plasma
from pic.particles import particles

from pic.functions import generate_maxw, velocity_maxw_flux, max_vect, fux_vect, numba_return_density, smooth
from pic.constantes import (me, q,kb,eps_0,mi)

#parameters
Lx = 1e-2 #System length
dX = 1e-5 #dX in m
Nx = int(Lx/dX)+1 #cell number
Lx = Nx*dX
print("Nx = {Nx}, and Lx = {Lx} cm".format(Nx = Nx, Lx = Lx*100))

Npart = 50*Nx #particles number, in particle par cell
n = 3e17  #[m^-3]
dT = 4e-12 #time step
Te = 30;     #[eV] Electron distribution temperature
Ti = 5 #[eV]

L_De = np.sqrt(eps_0*Te/(q*n))

dT = 1e-10 #time step

pla = plasma(dT,Nx,Lx,Npart,n,Te,Ti)
print(len(pla.ele.x),len(pla.ion.x))

#%%snakeviz
pla = plasma(dT,Nx,Lx,Npart,n,Te,Ti)

Nt = 2000
plt.plot(smooth(pla.phi),label = "init");

for nt in np.arange(Nt):
    if nt%10 == 0 : print("\r t = {:2.4f} over {:2.4f} mu s".format(nt*pla.dT*1e6,Nt*pla.dT*1e6),end="")
    #print("pusher")
    pla.pusher()
    #print("bound")
    pla.boundary()
    #print("rho")
    pla.compute_rho()
    #print("poisson")
    pla.solve_poisson()
    #print(len(pla.ele.x),len(pla.ion.x))

plt.plot(smooth(pla.phi),label = "Phi (end)");

plt.legend()
plt.show()


plt.plot(smooth(pla.ne),label = "end elecs");
plt.plot(smooth(pla.ni),label = "end ions");
plt.legend()
print(pla.ne.sum()*Lx,pla.ni.sum()*Lx)
plt.show()

plt.hist(pla.ele.V[:,0],bins=50, alpha = 0.7,density=True);
plainit = plasma(dT,Nx,Lx,Npart,n,Te,Ti)
plt.hist(plainit.ele.V[:,0],bins=50, alpha = 0.7,density=True);

plt.show()

plt.plot(pla.history["Ie_w"], label = "Ie")
plt.plot(pla.history["Ii_w"], label = "Ii")
plt.legend()
plt.show()

ve = np.zeros(Nx)
parts = pla.ion
for i in np.arange(Nx):
    ve[i] = np.mean(parts.V[(parts.x > pla.x_j[i]) & (parts.x < pla.x_j[i]+pla.dx),0])
plt.plot(pla.x_j[:-1],ve)
plt.show()
