from __future__ import annotations
import itertools
import numpy as np
from scipy.integrate import simpson

from .utils import gaunt
from .ylmexpansion import YlmExpansion

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

def build_l2_core_states():
    return [ YlmExpansion(l=1, data= { (+1,0) : 1.0 } ),
                    YlmExpansion(l=1, data= { (+1,+1) : 1.0/np.sqrt(3), (0,0) : np.sqrt(2)/np.sqrt(3) }),
                    YlmExpansion(l=1, data= { (-1,0) : 1.0/np.sqrt(3.), (0,+1): np.sqrt(2)/np.sqrt(3) }),
                    YlmExpansion(l=1, data= { (-1,+1) : 1.0} )
                  ]


def _dipole(core_state:YlmExpansion, state:YlmExpansion, polarization:YlmExpansion) -> complex:
    spin_flip = lambda m : -1 if abs(m) == 1 else 1
    result = 0.0+0.0j
    for (m_d, coeff_d) in state:
        for (m_c, spin_c, coeff_c) in core_state:
            for (m_q, coeff_q) in polarization:
                result += spin_flip(m_c)*coeff_c*coeff_d*coeff_q*gaunt(m_c, m_q, m_d)
    return result

def _initial_to_final_transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          final:YlmExpansion, 
                          incoming_pol:YlmExpansion, 
                          outgoing_pol:YlmExpansion) -> float:
    return abs( sum([np.conj(_dipole(core_state, final, outgoing_pol))*_dipole(core_state, initial, incoming_pol) for core_state in core_states]) )**2

def _transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          pol:YlmExpansion) -> float:
    return abs( sum(  [_dipole(core_state, initial, pol) for core_state in core_states]) )**2


def rixs_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], incoming_pol:YlmExpansion, outgoing_pols:list[YlmExpansion]|YlmExpansion ) -> np.ndarray:
    outgoing_pols = [outgoing_pols] if not isinstance(outgoing_pols, list) else outgoing_pols
    dim = len(states)
    M = np.zeros((dim, dim), dtype=float)
    for pol in outgoing_pols: 
        for initial in range(dim):
            for final in range(dim):
                M[initial, final] += _initial_to_final_transition_amplitude(core_states, states[initial], states[final], incoming_pol, pol)
    return M

def xas_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], polarization:YlmExpansion):
    dim = len(states)
    M = np.zeros(dim, dtype=float)
    for initial in range(dim):
        M[initial] = _transition_amplitude(core_states, states[initial], polarization)
    return M

def rixs_cross_section(e_mesh:np.ndarray, 
                       density_of_states:np.ndarray, 
                       pol_matrix_elements:np.ndarray|None=None, 
                       Gamma:float = 0.6, 
                       Emin:float=0.0, 
                       Emax:float=10.0) -> np.ndarray:

    dim_states = density_of_states.shape[-1]

    pol_matrix_elements = pol_matrix_elements if pol_matrix_elements is not None else np.eye(dim_states) # check for correctness

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
            lorentz = 0.5*Gamma / ( (eocc - eout)**2 + 0.25*Gamma**2 )
            rho_shifted = np.stack([np.interp(e_interp, e_mesh[above_zero], density_of_states[above_zero, initial], left=0.0, right=0.0) 
                                     for initial in range(dim_states)], axis=1)
            integrand = np.einsum('ei,ef,e->eif', rho_shifted, rho_occ, lorentz)
            total = np.einsum('eif,if->e', integrand, pol_matrix_elements)
            cross_section[ie,je] = simpson(total, x=eocc)
    return x_grid, y_grid, cross_section

def xas(e_mesh:np.ndarray, 
                       density_of_states:np.ndarray, 
                       pol_matrix_elements:np.ndarray,
                       Gamma:float = 0.6, 
                       Emin:float=0.0, 
                       Emax:float=10.0) -> np.ndarray:

    dim_states = density_of_states.shape[-1]

    #pol_matrix_elements = pol_matrix_elements if pol_matrix_elements is not None else np.eye(dim_states) # check for correctness

    below_zero = np.where(e_mesh < 0.0)[0]
    above_zero = np.where(e_mesh > 0.0)[0]

    eunocc = e_mesh[above_zero]
    rho_unocc = density_of_states[above_zero, :]

    Ein   = e_mesh[(e_mesh > Emin) & (e_mesh < Emax)]

    xas_data = np.zeros_like(Ein, dtype=float)

    for (ie, ein) in enumerate(Ein):
        lorentz = 0.5*Gamma / ( (eunocc - ein)**2 + 0.25*Gamma**2 )
        for state in range(dim_states):
            integrand    = rho_unocc[:, state]*pol_matrix_elements[state]*lorentz
            xas_data[ie] += simpson(integrand, x=eunocc)
    return Ein, xas_data
