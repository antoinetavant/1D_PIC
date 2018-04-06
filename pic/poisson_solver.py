"""Solver of the Poisson equation for a 1D PIC simulation"""


import numpy as np
from functions import numba_thomas_solver


class Poisson_Solver(object):
    """docstring for Poisson_Solver."""
    def __init__(self, Nx, Boundary):
        super(Poisson_Solver, self).__init__()
        self.Nx = Nx
        self.Boundary = Boundary


    def solve(rho):
        """general solver, for Thomas and SOR"""
        pass



    def init_thomas(self):
        """Initialisation of the {b, a, c}_i values, and {c'}_i

        """

        self.bi = - np.ones(self.Nx+1, dtype = 'float')
        self.bi[1:-1] *= 2

        [self.ai, self.ci ]= np.ones((2,self.Nx+1), dtype = 'float')
        self.ci[[0,-1]] = 0.
        if True:
            #Wall at the right also
            self.ai[[0,-1]] = 0.

        else:
            #neuman at right boundary condition
            self.ai[0] = 0.

        ciprim = np.copy(self.ci) #copy the value, not the reference
        ciprim[0] /= self.bi[0]
        for i in np.arange(1,len(ciprim)):
         ciprim[i] /= self.bi[i] - self.ai[i]*ciprim[i-1]

        self.ciprim = ciprim
        self.inited_thomas = True


    def thomas_solver(self, rho, dx = 1., q = 1., qf = 1., eps_0 = 1.):
        """solve phi for Rho using Thomas solver, need initialisation first
        """

        try:
            self.inited_thomas
        except NameError:
            #Need to define thomas Parameters
            self.init_thomas

        # Boundary configuration
        rho[[0,-1]] = 0

        #RHS
        di = - rho.copy()*dx/(q*qf)
        di[0] = 0 /(dx) #Boundary condition
        if(True):
            di[-1] = 0 /(dx) #Boundary condition

        phi = numba_thomas_solver(di,self.ai, self.bi, self.ciprim,self.Nx)

        phi *= eps_0/(q*qf)
         #        #Poisson finished



        E = - np.gradient(phi, dx)

        return phi
