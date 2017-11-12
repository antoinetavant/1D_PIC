

import numpy as np
import scipy as sp
import astropy

from imp import reload
from .plasma import plasma as pl
from .functions import generate_maxw, velocity_maxw_flux


me = 9.109e-31; #[kg] electron mass
q = 1.6021765650e-19; #[C] electron charge
kb = 1.3806488e-23;  #Blozman constant
eps_0 = 8.8548782e-12; #Vaccum permitivitty
mi = 131*1.6726219e27 #[kg]


class particles:
    """a Class with enouth attribute and methode to deal with particles"""
    def __init__(self, Npart,T,m,pl ):

        self.T = T
        self.m = m
        self.Npart = Npart
        self.x = np.zeros(Npart)
        self.V = np.zeros(Npart)*3

        self.pl = pl

        self.init_part()

    def init_part(self):
        """Generate uniforme particle, with maxwellian stuff"""

        from random import random
        self.x = np.array([random()*self.pl.Lx for i in np.arange(self.Npart)])

        self.V = [[generate_maxw(self.T,self.m),generate_maxw(self.T,self.m),generate_maxw(self.T,self.m)] for i in np.arange(self.Npart)]
        self.V = np.array(self.V)

    def add_uniform_part(self):
        """Generate one uniforme particle, with maxwellian stuff"""

        from random import random
        self.x = np.append(self.x,random()*self.pl.Lx)

        self.V = np.append(self.V,
                           [[generate_maxw(self.T,self.m),generate_maxw(self.T,self.m),generate_maxw(self.T,self.m)]],
                           axis=0)

    def add_flux_part(self):
        """Generate one particle, with maxwellian flux velocity.

        Position of the particle is with respect to 0, in the positive direction """

        from random import random
        self.V = np.append(self.V,
                           [[velocity_maxw_flux(self.T,self.m),velocity_maxw_flux(self.T,self.m),velocity_maxw_flux(self.T,self.m)]],
                           axis=0)


        self.x = np.append(self.x,random()*self.V[-1,0]*self.pl.dT)
