from __future__ import annotations 
import numpy as np
from scipy.special import factorial as fact
from sympy.physics.quantum.cg import CG

def print_matrix(A:np.ndarray)->None:
    for row in A:
        fmt = '{:9.5f} '*len(row)
        print(fmt.format(*row) )

def three_j_symbol(j1:int, m1:int, j2:int, m2:int, j3:int, m3:int) -> float:
    """_summary_

    :param j1: _description_
    :type j1: int
    :param m1: _description_
    :type m1: int
    :param j2: _description_
    :type j2: int
    :param m2: _description_
    :type m2: int
    :param j3: _description_
    :type j3: int
    :param m3: _description_
    :type m3: int
    :return: _description_
    :rtype: float
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
    """_summary_

    :param m1: _description_
    :type m1: int
    :param m2: _description_
    :type m2: int
    :param m3: _description_
    :type m3: int
    :return: _description_
    :rtype: float
    """
    coeff = np.sqrt(45.0/np.arctan(1.0)/16.0)
    a = three_j_symbol(l1, 0, l2, 0, l3, 0)
    b = three_j_symbol(l1, m1, l2, m2, l3, m3)
    return coeff*a*b

def gaunt_sympy(l1:float, m1:float, l2:float, m2:float, l3:float, m3:float) -> complex:
    return complex(CG(l1, m1, l2, m2, l3, m3).doit())