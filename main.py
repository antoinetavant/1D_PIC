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

import pickle

#parameters
Lx = 1e-2 #System length
dX = 0.7e-5 #dX in m
Nx = int(Lx/dX)+1 #cell number
Lx = Nx*dX
print("Nx = {Nx}, and Lx = {Lx} cm".format(Nx = Nx, Lx = Lx*100))

Npart = 50*Nx #particles number, calculated via particle par cell
n = 3e17  #[m^-3]
dT = 3e-12 #time step
Te_0 = 30;     #[eV] Electron distribution temperature
Ti_0 = 5 #[eV]

L_De = np.sqrt(eps_0*Te_0/(q*n))
wpe = np.sqrt(n*q**2/(eps_0*me))

print(f"L_de = {L_De*1e3:2.3f} mm, dX = {dX*1e3} mm")
print(f"time step dT = {dT*1e12:2.2f} mu s, wpe = {wpe**(-1)*1e12:2.2f} mus")

pla = plasma(dT,Nx,Lx,Npart,n,Te_0,Ti_0)
if not pla.v:
    exit()

Nt = 20000
pla.Do_diags = True
pla.n_0 = 0#int(Nt/2)
pla.n_a = 200

#  _______   ______       __        ______     ______   .______     _______.
# |       \ /  __  \     |  |      /  __  \   /  __  \  |   _  \   /       |
# |  .--.  |  |  |  |    |  |     |  |  |  | |  |  |  | |  |_)  | |   (----`
# |  |  |  |  |  |  |    |  |     |  |  |  | |  |  |  | |   ___/   \   \
# |  '--'  |  `--'  |    |  `----.|  `--'  | |  `--'  | |  |   .----)   |
# |_______/ \______/     |_______| \______/   \______/  | _|   |_______/
#

for nt in np.arange(Nt):

    if nt%10 == 0 : print("\r t = {:2.5f} over {:2.5f} mu s".format(nt*pla.dT*1e6,Nt*pla.dT*1e6),end="")
    pla.pusher()
    pla.boundary()
    pla.compute_rho()
    pla.solve_poisson()
    pla.diags(nt)

#
#      _______.     ___   ____    ____  _______     _______       ___   .___________.    ___
#     /       |    /   \  \   \  /   / |   ____|   |       \     /   \  |           |   /   \
#    |   (----`   /  ^  \  \   \/   /  |  |__      |  .--.  |   /  ^  \ `---|  |----`  /  ^  \
#     \   \      /  /_\  \  \      /   |   __|     |  |  |  |  /  /_\  \    |  |      /  /_\  \
# .----)   |    /  _____  \  \    /    |  |____    |  '--'  | /  _____  \   |  |     /  _____  \
# |_______/    /__/     \__\  \__/     |_______|   |_______/ /__/     \__\  |__|    /__/     \__\


pla.save_data()
