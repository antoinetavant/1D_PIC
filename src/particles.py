

import numpy as np
import scipy as sp
import astropy

from imp import reload
from functions import generate_maxw, max_vect, fux_vect, numba_return_density
from numpy.random import rand

from numba import jit

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
        self.x = np.zeros(Npart, dtype='float64')
        self.V = np.zeros(Npart, dtype='float64')*3

        self.pl = pl

        self.init_part()

    def init_part(self):
        """Generate uniforme particle, with maxwellian stuff"""

        from random import random
        from numpy.random import rand
        self.x = rand(self.Npart)*self.pl.Lx

        self.V = [max_vect(self.Npart,self.T,self.m),
                  max_vect(self.Npart,self.T,self.m),
                  max_vect(self.Npart,self.T,self.m)]

        self.V = np.array(self.V).reshape((self.Npart,3))

    def add_uniform_vect(self,N):
        """Generate one uniforme particle, with maxwellian stuff"""

        if N > 0:
            self.x = np.append(self.x,rand(N)*self.pl.Lx)

            self.V = np.append(self.V,
                               np.array([max_vect(N,self.T,self.m),
                                max_vect(N,self.T,self.m),
                                max_vect(N,self.T,self.m)]).T,
                                axis=0)

    def add_flux_vect(self,N):
        """ add N particle as flux ion the X direction"""
        if N > 0:
            self.V = np.append(self.V,
                                np.array([-fux_vect(N,self.T, self.m),
                                max_vect(N,self.T, self.m),
                                max_vect(N,self.T, self.m)]).T,
                                axis=0)

            self.x = np.append(self.x,self.pl.Lx + rand(N)*self.V[-N:,0]*self.pl.dT)



    def return_density(self,tabx):
        """interpolate the density """

        n = np.zeros(len(tabx),dtype='float64')

        return numba_return_density(int(self.Npart), self.x, tabx, n, self.pl.dx)

    def returnindex(self,x):
        """return the index of the cell where the particle is"""
        return int(x/self.pl.dx)
