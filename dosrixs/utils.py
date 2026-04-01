from __future__ import annotations 
from functools import lru_cache
import numpy as np
from scipy.special import factorial as fact

def print_matrix(A:np.ndarray)->None:
    for row in A:
        fmt = '{:9.5f} '*len(row)
        print(fmt.format(*row) )

def three_j_symbol(j1:int, m1:int, j2:int, m2:int, j3:int, m3:int) -> float:
    """Internal function to compute Wigner-3j.
    """
    if (m1+m2+m3 != 0 or
        m1 < -j1 or m1 > j1 or
        m2 < -j2 or m2 > j2 or
        m3 < -j3 or m3 > j3 or
        j3 > j1 + j2 or
        j3 < abs(j1-j2)):
        return 0.0
    three_j_sym = -1.0 if (j1-j2-m3) % 2 else 1.0
    three_j_sym *= np.sqrt(fact(j1+j2-j3)*fact(j1-j2+j3)*fact(-j1+j2+j3)/fact(j1+j2+j3+1))
    three_j_sym *= np.sqrt(fact(j1-m1)*fact(j1+m1)*fact(j2-m2)*fact(j2+m2)*fact(j3-m3)*fact(j3+m3))
    t_sum = sum ( [(-1.0 if t % 2 else 1.0)/(fact(t)*fact(j3-j2+m1+t)*fact(j3-j1-m2+t)*fact(j1+j2-j3-t)*fact(j1-m1-t)*fact(j2+m2-t)) 
                   for t in range(max(j2-j3-m1,j1-j3+m2,0),min(j1-m1,j2+m2,j1+j2-j3)+1)])
    three_j_sym *= t_sum
    return float(three_j_sym)

@lru_cache
def gaunt(l1:int, m1:int, l2:int, m2:int, l3:int, m3:int) -> float:
    r"""Compute Gaunt coefficients:

    .. math:: G(l_{1}, l_{2}, l_{3}, m_{1}, m_{2}, m_{3}) = \int d \Omega Y_{l_{1}}^{m_{1}}(\Omega)Y_{l_{2}}^{m_{2}}(\Omega)Y_{l_{3}}^{m_{3}}(\Omega) 

    :param l1: angular quantum number
    :type l1: int
    :param m1: magnetic quantum number
    :type m1: int
    :param l2: angular quantum number
    :type l2: int
    :param m2: magnetic quantum number
    :type m2: int
    :param l3: angular quantum number
    :type l3: int
    :param m3: magnetic quantum number
    :type m3: int
    :return: Gaunt coefficient
    :rtype: float
    """
    coeff = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    a = three_j_symbol(l1, 0, l2, 0, l3, 0)
    b = three_j_symbol(l1, m1, l2, m2, l3, m3)
    return coeff*a*b

def gaunt_sympy(l1:float, m1:float, l2:float, m2:float, l3:float, m3:float) -> complex:
    """Wrapper around sympy.physics.quantum.cg."""
    from sympy.physics.quantum.cg import CG
    return complex(CG(l1, m1, l2, m2, l3, m3).doit())