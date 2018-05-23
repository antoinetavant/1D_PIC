

import numpy as np

import pickle

from particles import particles
from functions import (numba_interp1D, generate_maxw,numba_interp1D_normed,
                       velocity_maxw_flux, numba_return_part_diag)
from numba import jit
from constantes import(me, q,kb,eps_0,mi)

from poisson_solver import Poisson_Solver

import h5py

class plasma:
    """a class with fields, parts, and method to pass from one to the other"""

    def __init__(self,dT,Nx,Lx,Npart,n,Te,Ti, n_average = 1, n_0 = 0, Do_diags = True):

        #Parameters
        self.Lx = Lx
        self.Nx = Nx
        self.N_cells = self.Nx +1
        self.dT = dT
        self.n = n
        self.qf = self.n*self.Lx/(Npart)
        self.n_0 = n_0
        self.n_average = n_average


        #poisson Solver
        self.PS = Poisson_Solver(self.N_cells, [0])
        self.PS.init_thomas()

        #Simulations
        self.ele = particles(Npart,Te,me,self)
        self.ion = particles(Npart,Ti,mi,self)

        self.E = np.zeros((self.N_cells,3))
        self.phi = np.zeros(self.N_cells)
        self.ne = np.zeros((self.N_cells))
        self.ni = np.zeros((self.N_cells))
        self.rho = np.zeros((self.N_cells))

        self.Do_diags = Do_diags

        self.history = {'Ie_w' : [],
                        'Ii_w' : [],
                        'Ie_c' : [],
                        'Ii_c' : []
                        }
        self.data = {}

        self.x_j = np.arange(0,self.N_cells, dtype='float64')*self.Lx/(self.N_cells)
        self.Lx = self.x_j[-1]
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

        for sign,part in zip([-1,1],[self.ele,self.ion]):

            for i in directions:
                #fast calculation
                try:
                    x_mesh = (self.x_j/self.dx).astype(int)
                    Nmax = part.Npart - part.compt_out
                    partx = part.x[:Nmax]

                    vectE = numba_interp1D_normed(partx/self.dx,x_mesh,self.E[:,0])
                    part.V[:Nmax,i] += sign*q/part.m*self.dT*vectE
                except:
                    print(part.Npart , part.compt_out)
                    print(part.x,max(part.x),min(part.x),part.V[:,0],self.Lx)
                    raise ValueError

            part.x[:Nmax] += part.V[:Nmax,0] *self.dT

    def boundary(self):
        """look at the postition of the particle, and remove them if they are outside"""

        #wall absorbtion
        for key, part in zip(['Ie_w','Ii_w'],[self.ele,self.ion]):
            #(key)
            Nout = part.remove_parts(self.Lx)
            self.history[key].append(Nout)

        if True: #Boundary condition right : wall

            #Reinject the flux leaving the system
            Ncouple = min(self.history['Ie_w'][-1],
                          self.history['Ii_w'][-1])
            #Reinject to force the ion particle number constante
            Ncouple = self.ion.compt_out
            self.inject_particles(Ncouple,Ncouple)

        else: #Mirror refletion
        # THIS IS NOT WORKING !!!!!
            pass
            # for key, part in zip(['Ie_c','Ii_c'],[self.ele,self.ion]):
            #     #print(key)
            #     mask = self.get_sup(part.x , self.Lx)
            #     self.mirror_parts(part,mask)
            #
            #     self.history[key].append(np.count_nonzero(mask==1))
            #
            #     Ncouple = min(self.history['Ie_w'][-1],self.history['Ii_w'][-1])
            #     self.inject_particles(Ncouple,Ncouple)

    def mirror_parts(self,part,mask):
        '''function to measured performance'''
        part.x[mask] = 2*self.Lx - part.x[mask]
        part.V[mask,0] *= -1
        return

    def get_sup(self,partx, val):
        '''function to measured performance'''
        try:
            mask = (partx > val)
        except RuntimeWarning:
            print(val,partx)

        return mask

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
        # Boundary configuration
        normed_rho[[0,-1]] = 0

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
            if (nt - self.n_0) % self.n_average ==0 :
                self.Te,self.ve = np.zeros((2,self.N_cells))
                self.temp_ne,self.temp_ni,self.temp_phi, self.temp_rho = np.zeros((4,self.N_cells))
                self.n_diags = 0

            #do the diags
            self.n_diags += 1

            temp_ve = np.zeros(self.N_cells)
            Nmax = parts.Npart - parts.compt_out -1
            partsx = parts.x[:Nmax]
            partV = parts.V[:Nmax,0]
            temp_ve = numba_return_part_diag(Nmax,
                                      partsx,
                                      partV,
                                      self.x_j,
                                      temp_ve,
                                      self.dx, power = 1)

            temp_Te = np.zeros(self.N_cells)
            temp_Te = numba_return_part_diag(Nmax,
                                      partsx,
                                      partV,
                                      self.x_j,
                                      temp_Te,
                                      self.dx, power = 2)

            old_setting = np.seterr(divide = "ignore")
            temp_Te /=  parts.Npart/self.N_cells # (self.ne)/( self.qf/self.dx)
            temp_ve /=  parts.Npart/self.N_cells #(self.ne)/( self.qf/self.dx)
            np.seterr(**old_setting)

            self.ve += temp_ve
            self.Te += (temp_Te - temp_ve**2 )*me/q
            self.temp_ne += self.ne
            self.temp_ni += self.ni
            self.temp_phi += self.phi
            self.temp_rho += self.ni - self.ne

            #Save data in dictionary if it is the last time step
            if np.mod(nt - self.n_0 +1 ,self.n_average) == 0 :
                tempdict = {"Te":self.Te,
                           "ne":self.temp_ne,
                           "ni":self.temp_ni,
                           "phi":self.temp_phi,
                           "ve":self.ve,
                           "rho":self.temp_rho,
                           }
                for k,v in tempdict.items():
                    tempdict[k] /= self.n_diags
                self.lastkey = str(nt)
                self.data[self.lastkey] = tempdict
                #del tempdict, temp_Te, temp_ve

    def save_data(self, filename = "data.dat"):
        """Save the data of the Simulations"""

        pickle.dump(self.data,open(filename,"wb"))


    def save_data_HDF5(self,filename = "data.h5", toopen = True):

        if toopen : self.f = h5py.File(filename, 'a')

        groupname = self.lastkey
        grp = self.f.create_group(groupname)

        data_dic = self.data[groupname]
        for k,v in data_dic.items():
            dset = grp.create_dataset(k, data = v)


    def create_filename(self,filename = "data", term = "h5"):
        """check the existence of a file.
         If it exists, change the name
         """
        import os.path
        if os.path.isfile(filename+"."+term):
            filename += "2"

        return filename + "." + term
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
