from __future__ import annotations 
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

def gaunt(l1:int=1, l2:int=1, l3:int=2, m1:int=0, m2:int=0, m3:int=0) -> float:
    r"""Compute Gaunt coefficients:

    .. math:: G(l_{1}, l_{2}, l_{3}, m_{1}, m_{2}, m_{3}) = \int d \Omega Y_{l_{1}}^{m_{1}}(\Omega)Y_{l_{2}}^{m_{2}}(\Omega)Y_{l_{3}}^{m_{3}}(\Omega) 

    :param l1: angular quantum number, defaults to 1
    :type l1: int, optional
    :param l2: angular quantum number, defaults to 1
    :type l2: int, optional
    :param l3: total angular momentum, defaults to 2
    :type l3: int, optional
    :param m1: magnetic quantum number, defaults to 0
    :type m1: int, optional
    :param m2: magnetic quantum number, defaults to 0
    :type m2: int, optional
    :param m3: total magnetic quanutm number, defaults to 0
    :type m3: int, optional
    :return: Gaunt coefficient
    :rtype: float
    """
    coeff = np.sqrt(45.0/np.arctan(1.0)/16.0)
    a = three_j_symbol(l1, 0, l2, 0, l3, 0)
    b = three_j_symbol(l1, m1, l2, m2, l3, m3)
    return coeff*a*b

def gaunt_sympy(l1:float, m1:float, l2:float, m2:float, l3:float, m3:float) -> complex:
    from sympy.physics.quantum.cg import CG
    """wrapper around sympy.physics.quantum.cg"""
    return complex(CG(l1, m1, l2, m2, l3, m3).doit())