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

from pic.gui import LivePlot

import pickle

#parameters
Lx = 1e-2 #System length
dX = 0.7e-5 #dX in m
Nx = int(Lx/dX) #cell number
Lx = Nx*dX
print("Nx = {Nx}, and Lx = {Lx} cm".format(Nx = Nx, Lx = Lx*100))

Npart = 10*Nx #particles number, calculated via particle par cell
n = 3e17  #[m^-3]
dT = 3e-12 #time step
Te_0 = 50;     #[eV] Electron distribution temperature
Ti_0 = 10 #[eV]

L_De = np.sqrt(eps_0*Te_0/(q*n))
wpe = np.sqrt(n*q**2/(eps_0*me))

print(f"L_de = {L_De*1e3:2.3f} mm, dX = {dX*1e3} mm")
print(f"time step dT = {dT*1e12:2.2f} mu s, wpe = {wpe**(-1)*1e12:2.2f} mus")

restartFileName = "data/restart_pla.dat"

try:
    pla = pickle.load(open(restartFileName,"rb"))
except:
    print("\n restart not found : initialisation   ")
    pla = plasma(dT,Nx,Lx,Npart,n,Te_0,Ti_0,n_average = 2000)

if not pla.v:
    exit()

Nt = int(2e-6/dT)
pla.Do_diags = True
pla.n_0 = 0#int(Nt/2)

doPlots = False

#  _______   ______       __        ______     ______   .______     _______.
# |       \ /  __  \     |  |      /  __  \   /  __  \  |   _  \   /       |
# |  .--.  |  |  |  |    |  |     |  |  |  | |  |  |  | |  |_)  | |   (----`
# |  |  |  |  |  |  |    |  |     |  |  |  | |  |  |  | |   ___/   \   \
# |  '--'  |  `--'  |    |  `----.|  `--'  | |  `--'  | |  |   .----)   |
# |_______/ \______/     |_______| \______/   \______/  | _|   |_______/
#

tabstr = ["phi", "Te","ni","ne", "ve", "rho"]
if doPlots:
    PlotObject = LivePlot(pla.x_j,tabstr )

    limites = { 'phi':[-10,150],
               "ni":[0,4.5e17],
               "ne":[0,4e17],
               "Te":[0,50],
               "ve":[-3e5,3e5],
               "rho":[-1e17, 2e17],
               }
    for ax, st in zip(PlotObject.axarr, tabstr):
        ax.set_ylim(*limites[st])

    plt.show()

dataFileName = pla.create_filename("data/run1","h5")



for nt in np.arange(Nt):
    pla.pusher()
    pla.boundary()
    pla.compute_rho()
    pla.solve_poisson()
    pla.diags(nt)

    if np.mod(nt - pla.n_0 +1 ,pla.n_average) == 0 :
        toopen = True if nt < pla.n_average else False
        pla.save_data_HDF5(dataFileName, True )

        #try:
        if doPlots: PlotObject.updatevalue(pla.data[ pla.lastkey], nt, Nt, dT)
        # except:
            # print(len(pla.x_j),len(pla.data[pla.lastkey]["phi"]),len(pla.data[pla.lastkey]["ni"]),len(pla.data[pla.lastkey]["ne"]))

        print("\r t = {:2.5f} over {:2.5f} mu s".format(nt*pla.dT*1e6,Nt*pla.dT*1e6),end="")
        #print("\r t = {} over {} mu s".format(len(pla.ele.x),len(pla.ion.x)),end="")

        if np.mod(nt - pla.n_0 +1 ,100*pla.n_average) == 0 : pickle.dump(pla, open(restartFileName,'wb'))

        pla.f.close()
#
#      _______.     ___   ____    ____  _______     _______       ___   .___________.    ___
#     /       |    /   \  \   \  /   / |   ____|   |       \     /   \  |           |   /   \
#    |   (----`   /  ^  \  \   \/   /  |  |__      |  .--.  |   /  ^  \ `---|  |----`  /  ^  \
#     \   \      /  /_\  \  \      /   |   __|     |  |  |  |  /  /_\  \    |  |      /  /_\  \
# .----)   |    /  _____  \  \    /    |  |____    |  '--'  | /  _____  \   |  |     /  _____  \
# |_______/    /__/     \__\  \__/     |_______|   |_______/ /__/     \__\  |__|    /__/     \__\


#pla.save_data("Data_both.dat")
