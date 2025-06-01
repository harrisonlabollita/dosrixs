from __future__ import annotations
from typing import Literal, cast
import numpy as np
from scipy.integrate import simpson

from .utils import gaunt, gaunt_sympy
from .ylmexpansion import YlmExpansion

# Define the d-orbitals in terms of YlmExpansions
DXY   = YlmExpansion(l=2, data= {(-2,0) : +1.0j/np.sqrt(2.0), (+2,0) : -1.0j/np.sqrt(2.0), (-2,1) : +1.0j/np.sqrt(2.0), (+2,1) : -1.0j/np.sqrt(2.0)})
DX2Y2 = YlmExpansion(l=2, data= {(-2,0) : +1.0/np.sqrt(2.0),  (+2,0) : +1.0/np.sqrt(2.0), (-2,1) : +1.0/np.sqrt(2.0),  (+2,1) : +1.0/np.sqrt(2.0)})
DZ2   = YlmExpansion(l=2, data= { (0,0) : +1.0, (0, 1) : +1.0 })
DYZ   = YlmExpansion(l=2, data= {(-1,0) : +1.0j/np.sqrt(2.0), (+1,0) : +1.0j/np.sqrt(2.0), (-1,1) : +1.0j/np.sqrt(2.0), (+1,1) : +1.0j/np.sqrt(2.0)})
DXZ   = YlmExpansion(l=2, data= {(-1,0) : +1.0/np.sqrt(2.0),  (+1,0) : -1.0/np.sqrt(2.0), (-1,1) : +1.0/np.sqrt(2.0),  (+1,1) : -1.0/np.sqrt(2.0) })

# Define electric-fields in terms of YlmExpansions for different surfaces
def build_electric_fields(normal:str = 'z') -> tuple[YlmExpansion, YlmExpansion, YlmExpansion]:
    """Construct the electric fields for a given geometry.

    :param normal: The surface normal direction, defaults to 'z'
    :type normal: str, optional
    :raises Exception: Throws an error for an incorrect normal surface
    :return: The electric fields for a chosen surface normal.
    :rtype: tuple[YlmExpansion, YlmExpansion, YlmExpansion]
    """
    if normal == 'z':
        EX = YlmExpansion(l=1, data = {(-1,0) : 1.0/np.sqrt(2),  (0,0) : 0.0, (+1,0) : -1.0/np.sqrt(2) })
        EY = YlmExpansion(l=1, data = {(-1,0) : 1.0j/np.sqrt(2), (0,0) : 0.0, (+1,0) : 1.0j/np.sqrt(2) })
        EZ = YlmExpansion(l=1, data = {(-1,0) : 0.0,             (0,0) : 1.0, (+1,0) : 0.0 })
        return EX, EY, EZ
    else:
        raise Exception(f"The surface normal = {normal} is not a valid option.")

DORBITAL = Literal["dxy", "dx2y2", "dz2", "dxz", "dyz"]
def _str2dorbital(d:DORBITAL) -> YlmExpansion:
    match d:
        case 'dxy':   return DXY
        case 'dx2y2': return DX2Y2
        case 'dz2':   return DZ2
        case 'dxz':   return DXZ
        case 'dyz':   return DYZ


def build_d_states(order:list[DORBITAL]=["dz2", "dxy", "dx2y2", "dxz", "dyz"]) -> list[YlmExpansion]:
    """The d-orbital valence states ordered by the order keyword.

    :param order: order of d-orbitals for different electronic structure codes, defaults to 'wien2k'
    :type order: str | list[str], optional
    :raises Exception: Throws an error if d-orbital does not match names.
    :return: the valence states corresponding to the d-orbtials
    :rtype: list[YlmExpansion]
    """
    return [ _str2dorbital(o) for o in order]
    # valid:list[str] = ['wien2k']
    # if order not in valid: raise Exception(f"The argument {order} is not a valid option! The valid options are {valid}")
    # if order == 'wien2k': return [DZ2, DXY, DX2Y2, DYZ, DXZ]

EDGE = Literal["L2", "L3"]

def get_core_edge_quantum_numbers(edge:EDGE) -> dict[str, float]:
    """Get the quantum number corresponding to an edge.

    :param edge: Core edge
    :type edge: EDGE
    :return: quantum numbers for edge
    :rtype: dict[str, float]
    """
    if   edge == "L2":  return {'n' : 2, 'l' : 1, 'j' : 0.5}
    elif edge == "L3": return {'n' : 2, 'l' : 1, 'j' : 1.5}

def build_core_states(edge:EDGE) -> list[YlmExpansion]: 
    """The core states for the a given RIXS edge process

    :param edge: The RIXS edge process, defaults to 'l2'
    :type edge: str, optional
    :return: the core states corresponding to the RIXS process given by edge.
    :rtype: list[YlmExpansion]
    """
    quantum_numbers = get_core_edge_quantum_numbers(edge)
    l, j, s = int(quantum_numbers['l']), quantum_numbers['j'], 0.5

    spin2idx = lambda x : 0 if x > 0 else 1

    core_states:list[YlmExpansion] = []
    for mj in [x for x in np.arange(-j, j+s, s) if x != 0]:
        data = {}
        for m in range(-l, l+1):
            for spin in [-s, s]:
                data[(m, spin2idx(spin))] = gaunt_sympy(l, m, s, spin, j, mj)
        core_states.append(YlmExpansion(l=l, data=data))
    return core_states

# computes the integral ∫ dΩ Y(lc, mc)*Y(l=1,mq)*Y(ld, md), which
# reduces to gaunt coefficients, G(mc, mq, md). 
def dipole(core_state:YlmExpansion, state:YlmExpansion, polarization:YlmExpansion) -> complex:
    r"""Compute

    .. math:: \langle c | \epsilon | d \rangle

    :param core_state: The core state |c⟩
    :type core_state: YlmExpansion
    :param state: The valence state |d⟩
    :type state: YlmExpansion
    :param polarization: The photon polarization ε
    :type polarization: YlmExpansion
    :return: ⟨c|ε|d⟩
    :rtype: complex
    """
    result = 0.0+0.0j
    for (m_d, _, coeff_d) in state:
        for (m_c, _, coeff_c) in core_state:
            for (m_q, _, coeff_q) in polarization:
                result += coeff_c*coeff_d*coeff_q*gaunt(m1=m_c,m2=m_q,m3=m_d)
    return result

# computes the transition amplitude | ∑c <f|ε'|c><c|ε|i>.|^2 
# where ε, ε' are the incoming and outgoing photon polarizations,
# |c> are the core states and |i>, |f> are the initial and 
# final valence states. 
def initial_to_final_transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          final:YlmExpansion, 
                          incoming_pol:YlmExpansion, 
                          outgoing_pol:YlmExpansion) -> float:
    r"""
    
    .. math:: \sum_{c,\epsilon'} \langle f | \epsilon' | c \rangle \langle c | \epsilon | i \rangle


    """
    return abs( sum([np.conj(dipole(core_state, final, outgoing_pol))*dipole(core_state, initial, incoming_pol) for core_state in core_states]) )**2

# computes the transition amplitude | ∑c <i|ε|c> |^2 
# where ε is the photon polarization, |c> are the core 
# states and |i> is valence state. 
def transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          pol:YlmExpansion) -> float:
    return abs( sum(  [dipole(core_state, initial, pol) for core_state in core_states]) )**2


def rixs_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], 
                         incoming_pols:list[YlmExpansion]|YlmExpansion, outgoing_pols:list[YlmExpansion]|YlmExpansion ) -> np.ndarray:
    """Compute the RIXS matrix elements at fixed incoming photon polarization and summed over outgoing photon polarizations.

    :param states: a list of valence states.
    :type states: list[YlmExpansion]
    :param core_states: a list of core states.
    :type core_states: list[YlmExpansion]
    :param incoming_pols: incoming photon polarizations.
    :type incoming_pols: list[YlmExpansion] | YlmExpansion
    :param outgoing_pols: a list of outgoing photon polarizations.
    :type outgoing_pols: list[YlmExpansion] | YlmExpansion
    :return: RIXS matrix elements for the initial and final valence states.
    :rtype: np.ndarray
    """

    incoming_pols = [incoming_pols] if not isinstance(incoming_pols, list) else incoming_pols
    outgoing_pols = [outgoing_pols] if not isinstance(outgoing_pols, list) else outgoing_pols

    dim_pols  = len(incoming_pols)
    dim_states = len(states)
    M = np.zeros((dim_pols, dim_states, dim_states), dtype=float)
    for (in_pol, incoming) in enumerate(incoming_pols):
        for outgoing in outgoing_pols: 
            for initial in range(dim_states):
                for final in range(dim_states):
                    M[in_pol, initial, final] += initial_to_final_transition_amplitude(core_states, states[initial], states[final], incoming, outgoing)
    return M

def xas_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], polarizations:list[YlmExpansion]) -> np.ndarray:
    """Computes the XAS matrix elements for a given polarization.

    :param states: a list of valence states
    :type states: list[YlmExpansion]
    :param core_states: a list of core states
    :type core_states: list[YlmExpansion]
    :param polarization: incoming photon polarizations
    :type polarization: list[YlmExpansion]
    :return: XAS matrix elements
    :rtype: np.ndarray
    """
    dim_pols = len(polarizations)
    dim_states = len(states)
    M = np.zeros((dim_pols, dim_states), dtype=float)
    for ip, pol in enumerate(polarizations):
        for initial in range(dim_states):
            M[ip, initial] = transition_amplitude(core_states, states[initial], pol)
    return M

def rixs_cross_section(e_mesh:np.ndarray, 
                       density_of_states:np.ndarray, 
                       pol_matrix_elements:np.ndarray|None=None, 
                       Gamma:float = 0.6, 
                       Emin:float=0.0, 
                       Emax:float=10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the RIXS spectra using the projected density of states and the polarization matrix elements.

    :param e_mesh: The energy grid (relative to the Fermi level) that the density of states is defined on.
    :type e_mesh: np.ndarray
    :param density_of_states: The orbitally-resolved density of states from an electronic structure calculation.
    :type density_of_states: np.ndarray
    :param pol_matrix_elements: RIXS polarization matrix elements
    :type pol_matrix_elements: np.ndarray
    :param Gamma: Core-hole broadening, defaults to 0.6
    :type Gamma: float, optional
    :param Emin: Lower bound on the incidient energy, defaults to 0.0
    :type Emin: float, optional
    :param Emax: Upper bound on the incidient energy, defaults to 10.0
    :type Emax: float, optional
    :return: incident energy grid, energy loss grid, and RIXS spectra
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    dim_states = density_of_states.shape[-1]

    pol_matrix_elements = pol_matrix_elements if pol_matrix_elements is not None else np.eye(dim_states).reshape(1, dim_states, dim_states) # check for correctness

    dim_pols = pol_matrix_elements.shape[0]
    dim_pol_mat = pol_matrix_elements.shape[1]
    assert dim_states == dim_pol_mat, "The dimension of the density of states is not equal to the number of polarization elements ({dim_states} != {dim_pol})"

    below_zero = np.where(e_mesh < 0.0)[0]
    above_zero = np.where(e_mesh > 0.0)[0]

    eocc = e_mesh[below_zero]
    rho_occ = density_of_states[below_zero, :]

    Ein   = e_mesh[(e_mesh > Emin) & (e_mesh < Emax)]
    Eloss = e_mesh[(e_mesh > 0.0) & (e_mesh < Emax)]

    x_grid, y_grid = np.meshgrid(Ein, Eloss)
    cross_section = np.zeros((dim_pols, len(Ein), len(Eloss)), dtype=float)

    for (ie, ein) in enumerate(Ein):
        for (je, eloss) in enumerate(Eloss):
            e_interp = eocc + eloss
            eout = ein - eloss
            lorentz = 0.5*Gamma / ( (eocc - eout)**2 + 0.25*Gamma**2 )
            rho_shifted = np.stack([np.interp(e_interp, e_mesh[above_zero], density_of_states[above_zero, initial], left=0.0, right=0.0) 
                                     for initial in range(dim_states)], axis=1)
            integrand = np.einsum('ei,ef,e->eif', rho_shifted, rho_occ, lorentz)
            for pol in range(dim_pols):
                total = np.einsum('eif,if->e', integrand, pol_matrix_elements[pol])
                cross_section[pol, ie,je] = simpson(total, x=eocc)
    return x_grid, y_grid, cross_section

def xas(e_mesh:np.ndarray, 
                       density_of_states:np.ndarray, 
                       pol_matrix_elements:np.ndarray,
                       Gamma:float = 0.6, 
                       Emin:float=0.0, 
                       Emax:float=10.0) -> tuple[np.ndarray, np.ndarray]:
    """Computes the x-ray absorption spectra (XAS) using the projected density of states and the polarization matrix elements.

    :param e_mesh: The energy grid (relative to the Fermi level) that the density of states is defined on.
    :type e_mesh: np.ndarray
    :param density_of_states: The orbitally-resolved density of states from an electronic structure calculation.
    :type density_of_states: np.ndarray
    :param pol_matrix_elements: XAS polarization matrix elements
    :type pol_matrix_elements: np.ndarray
    :param Gamma: Core-hole broadening, defaults to 0.6
    :type Gamma: float, optional
    :param Emin: Lower bound on the incidient energy, defaults to 0.0
    :type Emin: float, optional
    :param Emax: Upper bound on the incidient energy, defaults to 10.0
    :type Emax: float, optional
    :return: incident energy grid, x-ray absorption spectra
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    dim_states = density_of_states.shape[-1]
    dim_pol_el = pol_matrix_elements.shape[-1]
    assert dim_states == dim_pol_el, "The dimension of the density of states is not equal to the number of polarization elements ({dim_states} != {dim_pol})"

    dim_pol = pol_matrix_elements.shape[0]
    #pol_matrix_elements = pol_matrix_elements if pol_matrix_elements is not None else np.eye(dim_states) # check for correctness

    below_zero = np.where(e_mesh < 0.0)[0]
    above_zero = np.where(e_mesh > 0.0)[0]

    eunocc = e_mesh[above_zero]
    rho_unocc = density_of_states[above_zero, :]

    Ein   = e_mesh[(e_mesh > Emin) & (e_mesh < Emax)]

    xas_data = np.zeros((dim_pol, len(Ein)), dtype=float)

    for (ie, ein) in enumerate(Ein):
        lorentz = 0.5*Gamma / ( (eunocc - ein)**2 + 0.25*Gamma**2 )
        for state in range(dim_states):
            for pol in range(dim_pol):
                integrand    = rho_unocc[:, state]*pol_matrix_elements[pol, state]*lorentz
                xas_data[pol, ie] += simpson(integrand, x=eunocc)
    return Ein, xas_data
