from __future__ import annotations
from typing import Union, Iterator
from copy import deepcopy

Scalar = Union[int, float, complex]

_ATOL = 1e-12

class YlmExpansion(object):
    r"""An abstract representation of a state in a basis of spherical harmonics. A state is represented as:
    
    .. math::
        |\psi \rangle = \sum_{m,\sigma}c_{m\sigma}|l, m\rangle\otimes|\sigma\rangle

    :param l: angular quantum number
    :type l: int
    :param data: coefficients and basis functions represented as a dictionary
    :type data: dict[tuple[int,int],complex]
    """
    def __init__(self, l:int, data:dict[tuple[int,int],Scalar]):
        """Constructor method"""
        self._l:int = l
        self._data:dict[tuple[int,int],Scalar]  = data

    def __getitem__(self, key:tuple[int,int]) -> Scalar: return self._data.get(key, 0.0)

    def __iter__(self:YlmExpansion) -> Iterator[tuple[int, int, Scalar]]: 
        for x, y in self._data.items(): yield x[0], x[1], y

    def __repr__(self:YlmExpansion) -> str:  
        idx2spin = lambda x : '↑' if x == 0 else '↓'
        return " ".join([f"{val:.4f}*|{self._l},{key[0]}>⊗|{idx2spin(key[1])}>" for (key, val) in self._data.items() if abs(val) > _ATOL])
    
    def __eq__(self:YlmExpansion, x) -> bool: 
        if isinstance(x, YlmExpansion): 
            if self._l != x._l:
                return False
            all_keys = set(self._data.keys()).union(x._data.keys())
            return all(abs(self[key] - x[key]) < _ATOL for key in all_keys)
        raise ValueError(f"No equality operation available for {type(x).__name__}!")

    __str__ = __repr__

    # multiplication
    def __mul__(self,  x:Scalar)  -> YlmExpansion : return YlmExpansion(l=self._l, data={key : val*x for key, val in self._data.items() })
    def __rmul__(self, x:Scalar)  -> YlmExpansion : return YlmExpansion(l=self._l,  data={key : val*x for key, val in self._data.items() })
    def __neg__(self) -> YlmExpansion : return -1*self

    # division
    def __truediv__(self, x:Scalar) -> YlmExpansion: return (1/x) * self 
    __itruediv__ = __truediv__

    # addition
    def __add__(self, x:YlmExpansion) -> YlmExpansion : 
        if not isinstance(x, YlmExpansion):
            raise TypeError(f"Can not add {type(self).__name__} with object of type {type(x).__name__}.")
        if self.angular_quantum_number != x.angular_quantum_number:
            raise ValueError(f"Can not add {type(self).__name__} with different angular quantum numbers ({self._l} != {x._l}).")
        return YlmExpansion(l=self._l, data={key : self[key] + x[key] for key in set(self._data.keys()).union(x._data.keys()) })

    # subtraction
    def __sub__(self, x:YlmExpansion) -> YlmExpansion : return self.__add__(-x)

    # copy
    def copy(self) -> YlmExpansion: return deepcopy(self)

    # conjugate
    def conjugate(self) -> YlmExpansion:
        r"""Return the complex conjugate of this expansion.

        :return: a new YlmExpansion with conjugated coefficients
        :rtype: YlmExpansion
        """
        return YlmExpansion(l=self._l, data={key : val.conjugate() if isinstance(val, complex) else val for key, val in self._data.items()})

    def norm(self) -> float:
        r"""Compute the norm :math:`\sqrt{\langle \psi | \psi \rangle} = \sqrt{\sum_{m,\sigma} |c_{m\sigma}|^2}`.

        :return: the norm of the state
        :rtype: float
        """
        return sum(abs(val)**2 for val in self._data.values())**0.5

    # properties

    @property
    def angular_quantum_number(self)->int: return self._l

    @property
    def magnetic_quantum_numbers(self)->list[int]: 
        return [m for m in range(-self._l, self._l+1) if any(abs(self._data.get((m, s), 0.0)) > _ATOL for s in range(2))]
