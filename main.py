#import
import numpy as np
import scipy as sp
import astropy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pic
from pic.plasma import plasma
from pic.particles import particles
from pic.functions import generate_maxw, velocity_maxw_flux, max_vect, fux_vect, numba_return_density, smooth
from pic.constantes import (me, q,kb,eps_0,mi)

import pickle

#parameters
Lx = 2e-2 #System length
dX = 0.7e-5 #dX in m
Nx = int(Lx/dX)+1 #cell number
Lx = Nx*dX
print("Nx = {Nx}, and Lx = {Lx} cm".format(Nx = Nx, Lx = Lx*100))

Npart = 100*Nx #particles number, calculated via particle par cell
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

Nt = int(2e-6/dT)
pla.Do_diags = True
pla.n_0 = 0#int(Nt/2)
pla.n_average = 500

#  _______   ______       __        ______     ______   .______     _______.
# |       \ /  __  \     |  |      /  __  \   /  __  \  |   _  \   /       |
# |  .--.  |  |  |  |    |  |     |  |  |  | |  |  |  | |  |_)  | |   (----`
# |  |  |  |  |  |  |    |  |     |  |  |  | |  |  |  | |   ___/   \   \
# |  '--'  |  `--'  |    |  `----.|  `--'  | |  `--'  | |  |   .----)   |
# |_______/ \______/     |_______| \______/   \______/  | _|   |_______/
#



plt.ion() ## Note this correction
fig=plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 4)

ax1.set_title("Potential")
ax2.set_title("ne")
ax3.set_title("ni")

line1 = Line2D([], [], color='black')
line2 = Line2D([], [], color='red', linewidth=2)
line3 = Line2D([], [], color='red')
ax1.add_line(line1)
ax2.add_line(line2)
ax3.add_line(line3)
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(pla.x_j[0], pla.x_j[-1])

ax1.set_ylim(-10,150)
ax2.set_ylim(0,5e17)
ax3.set_ylim(0,4e17)

for nt in np.arange(Nt):
    pla.pusher()
    pla.boundary()
    pla.compute_rho()
    pla.solve_poisson()
    pla.diags(nt)

    if nt%int(pla.n_average/5) == 0 :

        line1.set_data(pla.x_j, pla.phi)
        line2.set_data(pla.x_j, pla.ne)
        line3.set_data(pla.x_j, pla.ni)

        plt.suptitle(f"Nt = {nt} over {Nt}, t = {nt*pla.dT*1e6:2.2e} $\mu s$", fontsize=14)
        plt.draw()
        plt.pause(0.00001) #Note this correction
        print("\r t = {:2.5f} over {:2.5f} mu s".format(nt*pla.dT*1e6,Nt*pla.dT*1e6),end="")
        #print("\r t = {} over {} mu s".format(len(pla.ele.x),len(pla.ion.x)),end="")


#
#      _______.     ___   ____    ____  _______     _______       ___   .___________.    ___
#     /       |    /   \  \   \  /   / |   ____|   |       \     /   \  |           |   /   \
#    |   (----`   /  ^  \  \   \/   /  |  |__      |  .--.  |   /  ^  \ `---|  |----`  /  ^  \
#     \   \      /  /_\  \  \      /   |   __|     |  |  |  |  /  /_\  \    |  |      /  /_\  \
# .----)   |    /  _____  \  \    /    |  |____    |  '--'  | /  _____  \   |  |     /  _____  \
# |_______/    /__/     \__\  \__/     |_______|   |_______/ /__/     \__\  |__|    /__/     \__\


pla.save_data("Data_both.dat")
