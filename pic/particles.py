

import numpy as np

from functions import (popout, max_vect, fux_vect,
                       numba_return_part_diag, mirror_vect)
from numpy.random import rand


class particles:
    """a Class with enouth attribute and methode to deal with particles"""

    def __init__(self, Npart, T, m, pl):

        self.T = T
        self.m = m
        self.Npart = int(Npart)
        self.x = np.zeros(Npart, dtype='float64')
        self.V = np.zeros(Npart, dtype='float64')*3

        self.pl = pl

        self.init_part()

    def init_part(self):
        """Generate uniforme particle, with maxwellian stuff"""

        from numpy.random import rand
        self.x = rand(self.Npart)*self.pl.Lx

        self.V = [max_vect(self.Npart, self.T, self.m),
                  max_vect(self.Npart, self.T, self.m),
                  max_vect(self.Npart, self.T, self.m)]

        self.V = np.array(self.V).reshape((self.Npart, 3))

        self.compt_out = 0

    def add_uniform_vect(self, N):
        """Generate one uniforme vector of particles,
         with maxwellian velocities"""

        if N > 0:
            if N <= self.compt_out:
                To_add = N
            if N > self.compt_out:
                To_add = self.compt_out

            # Fill the laast elemnts with the new ones
            Nmin = self.Npart - self.compt_out - 1
            Nmax = self.Npart - self.compt_out - 1 + To_add

            if True:
                x_tmp = rand(To_add)*self.pl.Lx
            else:
                x_tmp = np.ones(To_add, dtype="float")*self.pl.Lx/2
            self.x[Nmin:Nmax] = x_tmp

            self.V[Nmin:Nmax, :] = np.array([max_vect(To_add, self.T, self.m),
                                             max_vect(To_add, self.T, self.m),
                                             max_vect(To_add, self.T, self.m)]
                                            ).T

            N -= To_add
            self.compt_out -= To_add

        if N > 0:
            # we are adding to much particles : we need to extend the system
            self.Npart += N
            if True:
                x_tmp = rand(N)*self.pl.Lx
            else:
                x_tmp = np.ones(N, dtype="float")*self.pl.Lx/2
            self.x = np.append(self.x, x_tmp)

            self.V = np.append(self.V,
                               np.array([max_vect(N, self.T, self.m),
                                         max_vect(N, self.T, self.m),
                                         max_vect(N, self.T, self.m)]
                                        ).T, axis=0)

    def add_flux_vect(self, N):
        """ add N particle as flux ion the X direction"""
        if N > 0:
            self.V = np.append(self.V,
                               np.array([-fux_vect(N, self.T, self.m),
                                         max_vect(N, self.T, self.m),
                                         max_vect(N, self.T, self.m)]
                                        ).T, axis=0)

            self.x = np.append(self.x,
                               self.pl.Lx + rand(N)*self.V[-N:, 0]*self.pl.dT)

    def return_density(self, tabx):
        """interpolate the density """

        n = np.zeros_like(self.pl.rho)
        Nmax = self.Npart - self.compt_out - 1
        partx = self.x[: Nmax]
        density = numba_return_part_diag(len(partx), partx, partx,
                                         tabx, n, self.pl.dx, power=0)

        return density

    def returnindex(self, x):
        """return the index of the cell where the particle is"""
        return int(x/self.pl.dx)

    def remove_parts(self, Lx, bounds=["w", "w"]):
        """remove the pariclues thar are outside of the systeme, right or left

        The boundary arguments will be used to differe between wall and center
        (mirror)
        """

        total_out = popout(self.x[:-1-self.compt_out], self.V[:-1-self.compt_out], Lx)

        # this_out = total_out - self.compt_out

        self.compt_out += total_out

        return total_out

    def mirror_parts(self, Lx, bounds=["w", "w"]):
        """remove the pariclues thar are outside of the systeme, right or left

        The boundary arguments will be used to differe between wall and center
        (mirror)
        """

        iout = mirror_vect(self.x, self.V, Lx)

        return iout
