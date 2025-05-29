from __future__ import annotations

class YlmExpansion(object):

    def __init__(self, l:int, data=dict[int,complex]):
        self._l = l
        self._data = data

    def __getitem__(self, key:int|tuple) -> complex: return self._data.get(key,0.0)

    def __iter__(self):
        for x, y in self._data.items(): 
            if isinstance(x, tuple): yield x[0], x[1], y
            else: yield x, y

    def __repr__(self) -> str:
        return " ".join([f"+ {val}*Y({self._l},{key})" for (key, val) in self._data.items() ])

    __str__ = __repr__

    # multiplication
    def __mul__(self, x)  -> YlmExpansion : return YlmExpansion(l=self._l, data={key : val*x for key, val in self._data.items() })
    def __rmul__(self, x) -> YlmExpansion : return YlmExpansion(l=self._l,  data={key : val*x for key, val in self._data.items() })

    # addition
    def __add__(self, x:YlmExpansion) -> YlmExpansion : return YlmExpansion(l=self._l, data={key : self[key] + x[key] for key in set(self._data.keys()).union(x._data.keys()) })
    def __sub__(self, x:YlmExpansion) -> YlmExpansion : return YlmExpansion(l=self._l, data={key : self[key] - x[key] for key in set(self._data.keys()).union(x._data.keys()) })
