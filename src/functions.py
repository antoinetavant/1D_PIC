

import numpy as np
import scipy as sp
import astropy

me = 9.109e-31; #[kg] electron mass
q = 1.6021765650e-19; #[C] electron charge
kb = 1.3806488e-23;  #Blozman constant
eps_0 = 8.8548782e-12; #Vaccum permitivitty
mi = 131*1.6726219e27 #[kg]

def generate_maxw(T,m):
    import random
    v_Te = np.sqrt(q*T/m)

    W = 2


    while (W >= 1 or W <= 0):
        R1 = (random.random()*2 -1 )
        R2 = (random.random()*2 -1 )

        W = R1**2 + R2**2
    W = np.sqrt( (-2*np.log(W))/W)

    v = v_Te*R1*W
    return v

def velocity_maxw_flux(T,m):
    import random
    v_Te = np.sqrt(q*T/m)

    R = random.random()
    v = (v_Te*np.sqrt(-np.log(R)))
    return v
