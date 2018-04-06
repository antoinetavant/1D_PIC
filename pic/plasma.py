

import numpy as np
import scipy as sp
from scipy import interpolate
import astropy

import pickle

from functions import (numba_interp1D, generate_maxw,numba_interp1D_normed,
                       velocity_maxw_flux, numba_return_meanv, numba_return_stdv)
from numba import jit
from constantes import(me, q,kb,eps_0,mi)

from poisson_solver import Poisson_Solver


class plasma:
    """a class with fields, parts, and method to pass from one to the other"""

    def __init__(self,dT,Nx,Lx,Npart,n,Te,Ti, n_average = 1, n_0 = 0, Do_diags = True):

        from .particles import particles as particles

        #Parameters
        self.Lx = Lx
        self.Nx = Nx
        self.dT = dT
        self.qf = n*self.Lx/(Npart)
        self.n_0 = n_0
        self.n_average = n_0

        #poisson Solver
        self.PS = Poisson_Solver(self.Nx, [0])
        self.PS.init_thomas()

        #Simulations
        self.ele = particles(int(Npart*1),Te,me,self)
        self.ion = particles(Npart,Ti,mi,self)

        self.E = np.zeros((Nx+1,3))
        self.phi = np.zeros(Nx+1)
        self.ne = np.zeros((Nx))
        self.ni = np.zeros((Nx))
        self.rho = np.zeros((Nx+1))

        self.Do_diags = Do_diags

        self.history = {'Ie_w' : [],
                        'Ii_w' : [],
                        'Ie_c' : [],
                        'Ii_c' : []
                        }
        self.data = {}

        self.x_j = np.arange(0,Nx+1, dtype='float64')*self.Lx/(Nx)
        self.dx = self.x_j[1]

        self.init_poisson = False

        #Phisical constantes
        self.wpe = np.sqrt(n*q**2/(eps_0*me))
        self.LDe = np.sqrt(eps_0*Te/(q*n))
        #self.v = self.print_init()
        self.v = True
        if self.v:
            print("The initialisation as been validated  !!")
        else:
            print("The initialisation as not been validated  :'(")

    def print_init(self):
        """print some stuffs, upgrade would be a graphic interface
        """
        print("~~~~~~ Initialisation of Plasma simulation ~~~~~~~~~~")
        print(f"time step dT = {self.dT*1e12:2.2f} 10^-12 s, wpe = {(1/self.wpe)*1e12:2.2f} 10^-12 s")
        print(f"mesh step dX = {self.dx*1e6:2.2f} mu m, LDe = {self.LDe*1e6:2.2f}")
        print(f" Let's go !!")

        V = self.validated()

        return V

    def pusher(self):
        """push the particles"""
        directions = [0] #0 for X, 1 for Y, 2 for Z

        #Einterpol = [interpolate.interp1d(self.x_j,self.E[:,i]) for i in directions]
        #print("interpalation")
        #Einterpol = interpolate.interp1d(self.x_j,self.E[:,0])
        for sign,part in zip([-1,1],[self.ele,self.ion]):

            for i in directions:
                #fast calculation
                try:
                    x_mesh = (self.x_j/self.dx).astype(int)
                    vectE = numba_interp1D_normed(part.x/self.dx,x_mesh,self.E[:,0])
                    part.V[:,i] += sign*q/part.m*self.dT*vectE
                except:
                    print(part.x,max(part.x),min(part.x),part.V[:,0],self.Lx)
                    raise ValueError

            part.x += part.V[:,0] *self.dT

    def boundary(self):
        """look at the postition of the particle, and remove them if they are outside"""

        #wall absorbtion
        for key, part in zip(['Ie_w','Ii_w'],[self.ele,self.ion]):
            #(key)
            mask = self.get_sup(part.x, 0)
            self.keep_parts(part,mask)

            self.history[key].append(np.count_nonzero(mask==0))

        Ncouple = min(self.history['Ie_w'][-1],self.history['Ii_w'][-1])
        self.inject_particles(Ncouple,Ncouple)

        if True: #Boundary condition right : wall
            for key, part in zip(['Ie_c','Ii_c'],[self.ele,self.ion]):
                mask = self.get_inf(part.x,self.Lx)
                self.keep_parts(part,mask)

                self.history[key].append(np.count_nonzero(mask==0))

            Ncouple = min(self.history['Ie_c'][-1],self.history['Ii_c'][-1])
            self.inject_particles(Ncouple,Ncouple)

        else: #Mirror refletion
            for key, part in zip(['Ie_c','Ii_c'],[self.ele,self.ion]):
                #print(key)
                mask = self.get_sup(part.x , self.Lx)
                self.mirror_parts(part,mask)

                self.history[key].append(np.count_nonzero(mask==1))


    def mirror_parts(self,part,mask):
        '''function to measured performance'''
        part.x[mask] = 2*self.Lx - part.x[mask]
        part.V[mask,0] *= -1
        return

    def keep_parts(self,part,mask):
        '''function to measured performance'''
        part.x = part.x[mask]
        part.V = part.V[mask,:]
        return

    def get_sup(self,partx, val):
        '''function to measured performance'''
        return partx > val

    def get_inf(self,partx,val):
        '''function to measured performance'''
        return partx < val

    def inject_particles(self,Ne,Ni):
        """inject particle with maxwellian distribution uniformely in the system"""

        #Would be better to add an array of particle, not a single one
        self.ele.add_uniform_vect(Ne)

        self.ion.add_uniform_vect(Ni)

    def inject_flux(self,Ne,Ni):
        """inject particle with maxwellian distribution uniformely in the system"""

        #Would be better to add an array of particle, not a single one
        self.ele.add_flux_vect(Ne)
        self.ion.add_flux_vect(Ni)

    def compute_rho(self):
        """Compute the plasma density via the invers aera method"""

        #print("calculated density")
        self.ne = self.ele.return_density(self.x_j)
        self.ni = self.ion.return_density(self.x_j)

        #print("dernormalisation")
        denorm = self.qf/self.dx
        self.ni *= denorm
        self.ne *= denorm
        self.rho = self.ni - self.ne
        self.rho *= q


    def solve_poisson(self):
        """solve poisson via the Thomas method :
        A Phi = -rho/eps0
        """

        normed_rho = self.rho*self.dx/(q*self.qf)

        normed_phi = self.PS.thomas_solver( normed_rho, dx = 1., q = 1., qf = 1., eps_0 = 1.)

        self.phi = normed_phi*eps_0/(q*self.qf)
         #        #Poisson finished
        self.E[:,0] = - np.gradient(self.phi, self.dx)

    def diags(self, nt):
        """calculate the average of some values.

        """

        parts = self.ele

        if self.Do_diags and nt >= self.n_0 :
            #init averages
            if np.mod(nt - self.n_0,self.n_average) ==0 :
                self.Te,self.ve = np.zeros((2,self.Nx))
                self.temp_ne,self.temp_ni,self.temp_phi = np.zeros((3,self.Nx+1))
                self.n_diags = 0

            #do the diags
            self.n_diags += 1

            temp_ve = np.zeros(self.Nx)
            temp_ve = numba_return_meanv(len(parts.x),
                                      parts.x,
                                      parts.V[:,0],
                                      self.x_j,
                                      temp_ve,
                                      self.dx)

            temp_Te = np.zeros(self.Nx)
            temp_Te = numba_return_stdv(len(parts.x),
                                      parts.x,
                                      parts.V[:,0],
                                      self.x_j,
                                      temp_Te,
                                      self.dx)

            temp_Te /=  (0.1+self.ne[:-1])/( self.qf/self.dx)
            temp_ve /=  (0.1+self.ne[:-1])/( self.qf/self.dx)

            self.ve += temp_ve
            self.Te += (temp_Te - temp_ve**2 )*me/q

            self.temp_ne += self.ne
            self.temp_ni += self.ni
            self.temp_phi += self.phi

            #Save data in dictionary if it is the last time step
            if np.mod(nt - self.n_0 +1 ,self.n_average) == 0 :
                tempdict = {"Te":self.Te,
                           "ne":self.temp_ne,
                           "ni":self.temp_ni,
                           "phi":self.temp_phi,
                           "ve":self.ve}
                for k,v in tempdict.items():
                    tempdict[k] /= self.n_diags

                self.data[str(nt)] = tempdict
                del tempdict, temp_Te, temp_ve

    def save_data(self, filename = "data.dat"):
        """Save the data of the Simulations"""

        pickle.dump(self.data,open(filename,"wb"))



    def validated(self):
        from gui import GUI
        from tkinter import Tk

        root = Tk()
        #size of the window
        root.geometry("400x300")
        my_gui = GUI(root)
        str1=f"time step dT = {self.dT*1e12:2.2f} 10^-12 s, wpe = {(1/self.wpe)*1e12:2.2f} 10^-12 s"
        str2=f"mesh step dX = {self.dx*1e6:2.2f} mu m, LDe = {self.LDe*1e6:2.2f}"
        my_gui.add_text(str1)
        my_gui.add_text(str2)

        my_gui.add_button("ok",my_gui.ok)
        my_gui.add_button("not ok",my_gui.not_ok)

        root.mainloop()

        return my_gui.validated
