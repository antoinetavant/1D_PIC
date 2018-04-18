# test the values of the constantes
import pytest
import numpy as np



def test_constant_values():

    me_expected = 9.109e-31; #[kg] electron mass
    q_expected = 1.6021765650e-19; #[C] electron charge
    kb_expected = 1.3806488e-23;  #Blozman constant
    eps_0_expected = 8.8548782e-12; #Vaccum permitivitty
    mi_expected = 131*1.6726219e-27 #[kg]

    from constantes import(me, q,kb,eps_0,mi)

    assert np.isclose(me,me_expected)
    assert np.isclose(q,q_expected)
    assert np.isclose(kb,kb_expected)
    assert np.isclose(eps_0,eps_0_expected)
    assert np.isclose(mi,mi_expected)
