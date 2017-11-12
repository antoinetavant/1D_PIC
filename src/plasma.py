

import numpy as np
import scipy as sp
import astropy

from imp import reload
from . import particles
reload(particles)
from .particles import particles
from .functions import generate_maxw, velocity_maxw_flux

me = 9.109e-31; #[kg] electron mass
q = 1.6021765650e-19; #[C] electron charge
kb = 1.3806488e-23;  #Blozman constant
eps_0 = 8.8548782e-12; #Vaccum permitivitty
mi = 131*1.6726219e27 #[kg]


class plasma:
    """a class with fields, parts, and method to pass from one to the other"""

    def __init__(self,dT,Nx,Lx,Npart,n,Te,Ti):

        self.history = {'Ie_w' : [],
                        'Ii_w' : [],
                        'Ie_c' : [],
                        'Ii_c' : []
                        }

        self.Lx = Lx
        self.dT = dT
        self.qf = n*self.Lx/(Npart)
        self.ele = particles(Npart,Te,me,self)
        self.ion = particles(Npart,Ti,mi,self)

        self.E = np.zeros((Nx+1,3))
        self.phi = np.zeros(Nx+1)
        self.ne = np.zeros((Nx+1))
        self.ni = np.zeros((Nx+1))
        self.rho = np.zeros((Nx+1))

        self.x_j = np.arange(0,Nx+1)*self.Lx/(Nx)
        self.dx = self.x_j[1]

    def pusher(self):
        """push the particles"""
        from scipy import interpolate

        for sign,part in zip([-1,1],[self.ele,self.ion]):
            E = [interpolate.interp1d(self.x_j,self.E[:,i]) for i in [0,1,2]]
            for i in [0,1,2]:
                try:
                    part.V[:,i] -= sign*q/me*self.dT*E[i](part.x)
                except:
                    print(part.x,part.V[:,0],self.Lx)
                    raise ValueError
            part.x += part.V[:,0] *self.dT

        self.boundary()

    def boundary(self):
        """look at the postition of the particle, and remove them if they are outside"""

        for key, part in zip(['Ie_w','Ii_w'],[self.ele,self.ion]):
            mask = part.x < self.Lx
            part.x = part.x[mask]
            part.V = part.V[mask,:]
            self.history[key].append(np.count_nonzero(mask==0))

        for key, part in zip(['Ie_c','Ii_c'],[self.ele,self.ion]):
            mask = part.x >0
            part.x = part.x[mask]
            part.V = part.V[mask,:]
            self.history[key].append(np.count_nonzero(mask==0))

        self.inject_particles(self.history['Ie_w'][-1],self.history['Ii_w'][-1])

        self.inject_flux(self.history['Ie_c'][-1],self.history['Ii_c'][-1])

    def inject_particles(self,Ne,Ni):
        """inject particle with maxwellian distribution uniformely in the system"""

        #Would be better to add an array of particle, not a single one
        for i in np.arange(Ne):
            self.ele.add_uniform_part()

        for i in np.arange(Ni):
            self.ion.add_uniform_part()

    def inject_flux(self,Ne,Ni):
        """inject particle with maxwellian distribution uniformely in the system"""

        #Would be better to add an array of particle, not a single one
        for i in np.arange(Ne):
            self.ele.add_flux_part()
            self.ele.V[-1,0] *= -1
            self.ele.x[-1] *= -1
            self.ele.x[-1] += self.Lx

        for i in np.arange(Ni):
            self.ion.add_flux_part()
            self.ion.V[-1,0] *= -1
            self.ion.x[-1] *= -1
            self.ion.x[-1] += self.Lx

    def compute_rho(self):
        """Compute the plasma density via the invers aera method"""


        for n, part in zip([self.ne,self.ni],[self.ele,self.ion]):
            n[:] = 0
            for i in np.arange(part.Npart):
                try:
                    j = np.argwhere(self.x_j>=part.x[i])[0][0]
                except IndexError:
                    print(part.x[i])
                    j = Nx

                n[j-1] += (self.x_j[j] - part.x[i])
                n[j] += (part.x[i] - self.x_j[j-1])

        self.ni /= self.dx**2*self.qf; self.ne /= self.dx**2*self.qf
        self.rho = self.ni - self.ne
        self.rho *= q


    def solve_poisson(self):
        """solve poisson via the Thomas method :

        TO VALIDATE !!!!


        A Phi = -rho/eps0

        """

        di = self.rho/eps_0

        bi = - np.ones(self.Nx+1)
        bi[1:-2] *= 2

        ai, ci = np.ones(self.Nx+1), np.ones(self.Nx+1)

        ciprim = ci
        ciprim[0] /= bi[0]
        for i in np.arange(1,self.Nx+1):
            ciprim[i] /= bi[i] - ai[i]*ciprim[i-1]

        diprim = di
        diprim[0] /= bi[0]

        for i in np.arange(1,self.Nx+1):
            diprim[i] -= ai[i]*diprim[i-1]
            diprim[i] /= bi[i] - ai[i]*ciprim[i-1]

        self.phi[-1] = diprim[-1]
        for i in np.arange(self.Nx,0):
            self.phi[i] = diprim[i] - ciprim[i]*self.phi[i+1]

#        #Poisson finished
