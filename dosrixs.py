from __future__ import annotations
import itertools
import time
import numpy as np
from scipy.special import factorial as fact
from scipy.integrate import simpson

import matplotlib.pyplot as plt; plt.style.use('publish')

def print_matrix(A:np.ndarray)->None:
    for row in A:
        fmt = '{:9.5f} '*len(row)
        print(fmt.format(*row) )

def three_j_symbol(j1:int, m1:int, j2:int, m2:int, j3:int, m3:int) -> float:
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
    return three_j_sym


def clebsch_gordan(j1:int, m1:int, j2:int, m2:int, j3:int, m3:int) -> float:
    r"""
    Calculate the Clebsh-Gordan coefficient
    .. math::
       \langle j_1 m_1 j_2 m_2 | j_3 m_3 \rangle = (-1)^{j_1-j_2+m_3} \sqrt{2 j_3 + 1}
       \begin{pmatrix}
         j_1 & j_2 & j_3\\
         m_1 & m_2 & -m_3
       \end{pmatrix}.
    """
    norm = np.sqrt(2*j3+1)*(-1 if j1-j2+m3 % 2 else 1)
    return norm*three_j_symbol(j1, m1, j2, m2, j3, -m3)

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

def gaunt(m1:int, m2:int, m3:int) -> float:
    l1, l2, l3 = 1,1,2
    coeff = np.sqrt(45.0/np.arctan(1.0)/16.0)
    a = three_j_symbol(l1, 0, l2, 0, l3, 0)
    b = three_j_symbol(l1, m1, l2, m2, l3, m3)
    return coeff*a*b

if False:
    for m1 in [-1,0,1]:
        for m2 in [-1,0,1]:
            for m3 in [-2,-1, 0, 1, 2]:
                print(f"{m1:2d} {m2:2d} {m3:2d} {gaunt(m1, m2, m3):12.8f}")

# D-orbital definitions
DXY   = YlmExpansion(l=2, data= {-2 : +1.0j/np.sqrt(2.0), +2 : -1.0j/np.sqrt(2.0) })
DX2Y2 = YlmExpansion(l=2, data= {-2 : +1.0/np.sqrt(2.0),  +2 : +1.0/np.sqrt(2.0) })
DZ2   = YlmExpansion(l=2, data= { 0 : +1.0 })
DYZ   = YlmExpansion(l=2, data= {-1 : +1.0j/np.sqrt(2.0), +1 : +1.0j/np.sqrt(2.0) })
DXZ   = YlmExpansion(l=2, data= {-1 : +1.0/np.sqrt(2.0),  +1 : -1.0/np.sqrt(2.0) })

# Electric Field definitions
EX = YlmExpansion(l=1, data = {-1 : 1.0/np.sqrt(2),  0 : 0.0, +1 : 1.0/np.sqrt(2) })
EY = YlmExpansion(l=1, data = {-1 : 1.0j/np.sqrt(2), 0 : 0.0, +1 : 1.0j/np.sqrt(2) })
EZ = YlmExpansion(l=1, data = {-1 : 0.0,             0 : 1.0, +1 : 0.0 })

def _dipole(core_state:YlmExpansion, state:YlmExpansion, polarization:YlmExpansion) -> complex:
    spin_flip = lambda m : -1 if abs(m) == 1 else 1
    result = 0.0+0.0j
    for (m_d, coeff_d) in state:
        for (m_c, spin_c, coeff_c) in core_state:
            for (m_q, coeff_q) in polarization:
                result += spin_flip(m_c)*coeff_c*coeff_d*coeff_q*gaunt(m_c, m_q, m_d)
    return result

def _transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          final:YlmExpansion, 
                          incoming_pol:YlmExpansion, 
                          outgoing_pol:YlmExpansion) -> float:
    total_amp = 0.0+0.0j
    for core_state in core_states:
        total_amp += np.conj(_dipole(core_state, final, outgoing_pol))*_dipole(core_state, initial, incoming_pol)
    return abs(total_amp)**2


def dipole_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], incoming_pol:YlmExpansion, outgoing_pols:list[YlmExpansion]|YlmExpansion ) -> np.ndarray:
    outgoing_pols = [outgoing_pols] if not isinstance(outgoing_pols, list) else outgoing_pols
    dim = len(states)
    M = np.zeros((dim, dim), dtype=float)
    for pol in outgoing_pols: 
        for initial in range(dim):
            for final in range(dim):
                M[initial, final] += _transition_amplitude(core_states, states[initial], states[final], incoming_pol, pol)
    return M


def rixs_cross_section(e_mesh:np.ndarray, 
                       density_of_states:np.ndarray, 
                       pol_matrix_elements:np.ndarray|None = None, 
                       Gamma:float = 0.6, 
                       Emin:float=0.0, 
                       Emax:float=10.0) -> np.ndarray:

    dim_states = density_of_states.shape[-1]
    pol_matrix_elements = np.eye(dim_states) if pol_matrix_elements is None else pol_matrix_elements

    below_zero = np.where(e_mesh < 0.0)[0]
    above_zero = np.where(e_mesh > 0.0)[0]

    eocc = e_mesh[below_zero]
    rho_occ = density_of_states[below_zero, :]

    Ein   = e_mesh[(e_mesh > Emin) & (e_mesh < Emax)]
    Eloss = e_mesh[(e_mesh > 0.0) & (e_mesh < Emax)]

    x_grid, y_grid = np.meshgrid(Ein, Eloss)
    cross_section = np.zeros((len(Ein), len(Eloss)), dtype=float)

    for (ie, ein) in enumerate(Ein):
        for (je, eloss) in enumerate(Eloss):
            e_interp = eocc + eloss
            eout = ein - eloss
            #lorentz = 0.5*Gamma / ( (eocc - eout)**2 + 0.25*Gamma**2 )
            lorentz = 1.0 / ( (eocc - eout)**2 + 0.25*Gamma**2 )
            rho_shifted = np.stack([np.interp(e_interp, e_mesh[above_zero], density_of_states[above_zero, initial], left=0.0, right=0.0) 
                                     for initial in range(dim_states)], axis=1)
            integrand = np.einsum('ei,ef,e->eif', rho_shifted, rho_occ, lorentz)
            total = np.einsum('eif,if->e', integrand, pol_matrix_elements)
            cross_section[ie,je] = simpson(total, x=eocc)
    return x_grid, y_grid, cross_section

def get_density_of_states(filename:str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename)
    e = data[:,0]
    dos = 2*data[:,1:]
    dos[:,-1] /= 2.0
    dos[:,-2] /= 2.0
    return e, dos

def plot_cross_section(ax, data, **kwargs):
    ax.tick_params(which ='both', direction='out')
    img = ax.pcolormesh(data[-3], data[-2],data[-1].T, **kwargs)
    cbar = fig.colorbar(img, ax=ax, location='top', orientation='horizontal', shrink=0.7)
    return ax

#------------------ script starts -------------------------

# define orbital states
d_orbitals:list[YlmExpansion] = [DZ2, DXY, DX2Y2, DYZ, DXZ]


phi = np.deg2rad(180)
theta = np.deg2rad(15)
theta_prime = np.deg2rad(15-153)

s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
p_pol      = np.cos(theta)*EZ - np.sin(theta)*(np.cos(phi)*EX + np.sin(phi)*EY)             # p
pprime_pol = np.cos(theta_prime)*EZ - np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'

# define core states
core_states:list[YlmExpansion] = [ 
                                  YlmExpansion(l=1, data= { (+1,0) : 1.0 } ),
                                  YlmExpansion(l=1, data= { (+1,+1) : 1.0/np.sqrt(3), (0,0) : np.sqrt(2)/np.sqrt(3) }),
                                  YlmExpansion(l=1, data= { (-1,0) : 1.0/np.sqrt(3.), (0,+1): np.sqrt(2)/np.sqrt(3) }),
                                  YlmExpansion(l=1, data= { (-1,+1) : 1.0} )
                                  ]

# compute matrix elements \sum_ϵ' M_if(ϵ,ϵ')
print("--> computing s and p polarizations", end= " ")
start = time.perf_counter()
Ms : np.ndaarry = dipole_matrix_elements(d_orbitals, core_states, s_pol, [s_pol, pprime_pol])
Mp : np.ndaarry = dipole_matrix_elements(d_orbitals, core_states, p_pol, [s_pol, pprime_pol])
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

# compute RISX cross section using DOS and matrix elements
e_mesh, dos = get_density_of_states('ndnio2-3d-dos.lda.txt')

print('--> computing s cross section', end = " ")
start = time.perf_counter()
s_cross_section = rixs_cross_section(e_mesh, dos, Ms, Emin = -1.0, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )
start = time.perf_counter()
print('--> computing p cross section', end= " ")
p_cross_section = rixs_cross_section(e_mesh, dos, Mp, Emin = -1.0, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

from matplotlib.colors import LinearSegmentedColormap
def colormap(*args): return LinearSegmentedColormap.from_list('', list(args), N=256)

# plot results
vmax = np.max(s_cross_section[-1])
fig, ax = plt.subplots(1,2,sharey=True, figsize=(5,4))
for a in ax: a.set_xlabel(r"E$_{\text{in}}$ (eV)")
ax[0] = plot_cross_section(ax[0], s_cross_section,  vmin=0, vmax=vmax, cmap='rainbow')
ax[1] = plot_cross_section(ax[1], p_cross_section,  vmin=0, vmax=vmax, cmap='rainbow')
ax[0].set_ylabel(r"E$_{\text{loss}}$ (eV)")
plt.show()
