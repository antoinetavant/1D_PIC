"""Solver of the Poisson equation for a 1D PIC simulation"""


import numpy as np



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

        diprim = di.copy()   #copy the values, not the reference
        diprim[0] /= self.bi[0]

        for i in np.arange(1,len(diprim)):
            diprim[i] -= self.ai[i]*diprim[i-1]
            diprim[i] /= self.bi[i] - self.ai[i]*self.ciprim[i-1]

        #Init solution

        phi = np.zeros(self.Nx + 1)
        phi[-1] = diprim[-1]
        #limit conditions

        #SOLVE
        for i in np.arange(self.Nx-1,-1,-1):
            phi[i] = diprim[i] - self.ciprim[i]*phi[i+1]

        phi *= eps_0/(q*qf)
         #        #Poisson finished



        E = - np.gradient(phi, dx)

        return phi
