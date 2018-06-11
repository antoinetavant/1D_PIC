from particles import particles
from functions import (numba_interp1D, generate_maxw, numba_interp1D_normed,
                       velocity_maxw_flux, numba_return_part_diag)
from numba import jit
from constantes import (me, q, kb, eps_0, mi)

from plasma import plasma

from collections import namedtuple


class job:
    """a class that represent one job
    It is used to configure the parameters, launche, process, and so on.
    """
    knownParamers = ["Lx",
                     "dX",
                     "Nx",
                     "Lx",
                     "Npart",
                     "n",
                     "dT",
                     "Te_0",
                     "Ti_0",
                     "restartFileName",
                     "dataFileName",
                     ]

    def __init__(self, parameterDict):
        """initialisation"""
        self.readParameters(parameterDict)

    def readParameters(self, parameterDict):
        """read the parameterDict
        """

        for k in parameterDict.keys():
            if k not in self.knownParamers:
                raise NameError("Unknow parameter: {} is not in {}".format(
                    k, self.knownParamers))

        self.parameter = namedtuple("parameters", parameterDict.keys())(
            *parameterDict.values())
