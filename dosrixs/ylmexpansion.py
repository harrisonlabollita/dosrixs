from __future__ import annotations

class YlmExpansion(object):
    r"""An abstract representation of a state in a basis of spherical harmonics. A state is represented as:
    
    .. math::
        |\psi \rangle = \sum_{m,\sigma}c_{m\sigma}|l, m\rangle\otimes|\sigma\rangle

    :param l: angular quantum number
    :type l: int
    :param data: coefficients and basis functions represented as a dictionary
    :type data: dict[tuple[int,int],complex]
    """
    def __init__(self, l:int, data:dict[tuple[int,int],complex]):
        """Constructor method"""
        self._l:int = l
        self._data:dict[tuple[int,int],complex]  = data


    def __getitem__(self, key:tuple[int,int]) -> complex: return self._data.get(key, 0.0)

    def __iter__(self:YlmExpansion): 
        for x, y in self._data.items(): yield x[0], x[1], y

    def __repr__(self) -> str:  
        idx2spin = lambda x : '↑' if x > 0 else '↓'
        sign = lambda x : '+' if x.real > 0 else '-'
        return " ".join([f"{val:.4f}*|{self._l},{key[0]}>⊗|{idx2spin(key[1])}>" for (key, val) in self._data.items() if val !=0+0j])

    __str__ = __repr__

    # multiplication
    def __mul__(self,  x)  -> YlmExpansion : return YlmExpansion(l=self._l, data={key : val*x for key, val in self._data.items() })
    def __rmul__(self, x) -> YlmExpansion : return YlmExpansion(l=self._l,  data={key : val*x for key, val in self._data.items() })

    # addition
    def __add__(self, x:YlmExpansion) -> YlmExpansion : 
        assert self._l == x._l, f"Can not do arithmetic with YlmExpansions of different angular quantum numbers {self._l} != {x._l}"
        return YlmExpansion(l=self._l, data={key : self[key] + x[key] for key in set(self._data.keys()).union(x._data.keys()) })

    # subtraction
    def __sub__(self, x:YlmExpansion) -> YlmExpansion : 
        assert self._l == x._l, f"Can not do arithmetic with YlmExpansions of different angular quantum numbers {self._l} != {x._l}"
        return YlmExpansion(l=self._l, data={key : self[key] - x[key] for key in set(self._data.keys()).union(x._data.keys()) })

    @property
    def angular_quantum_number(self)->int: return self._l

    @property
    def magnetic_quantum_numbers(self)->list[int]: return list(range(-self._l, self._l+1))