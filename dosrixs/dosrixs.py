from __future__ import annotations
import itertools
import numpy as np
from scipy.integrate import simpson

from .utils import gaunt
from .ylmexpansion import YlmExpansion

# Define the d-orbitals in terms of YlmExpansions
DXY   = YlmExpansion(l=2, data= {-2 : +1.0j/np.sqrt(2.0), +2 : -1.0j/np.sqrt(2.0) })
DX2Y2 = YlmExpansion(l=2, data= {-2 : +1.0/np.sqrt(2.0),  +2 : +1.0/np.sqrt(2.0) })
DZ2   = YlmExpansion(l=2, data= { 0 : +1.0 })
DYZ   = YlmExpansion(l=2, data= {-1 : +1.0j/np.sqrt(2.0), +1 : +1.0j/np.sqrt(2.0) })
DXZ   = YlmExpansion(l=2, data= {-1 : +1.0/np.sqrt(2.0),  +1 : -1.0/np.sqrt(2.0) })

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
        EX = YlmExpansion(l=1, data = {-1 : 1.0/np.sqrt(2),  0 : 0.0, +1 : 1.0/np.sqrt(2) })
        EY = YlmExpansion(l=1, data = {-1 : 1.0j/np.sqrt(2), 0 : 0.0, +1 : 1.0j/np.sqrt(2) })
        EZ = YlmExpansion(l=1, data = {-1 : 0.0,             0 : 1.0, +1 : 0.0 })
        return EX, EY, EZ
    else:
        raise Exception(f"The surface normal = {normal} is not a valid option.")

# map strings to d-orbitals
def _str2dorbital(d:str) -> YlmExpansion: 
    valid:list[str] = ['dxy', 'dx2y2', 'dz2', 'dxz', 'dyz']
    x = d.lower()
    if x not in valid: RuntimeError(f"The argument {x} is not a valid option! The valid options are {valid}")
    match x:
        case 'dxy':   return DXY
        case 'dx2y2': return DX2Y2
        case 'dz2':   return DZ2
        case 'dxz':   return DXZ
        case 'dyz':   return DYZ


def build_d_states(order:str|list[str] = 'wien2k') -> list[YlmExpansion]:
    """The d-orbital valence states ordered by the order keyword.

    :param order: order of d-orbitals for different electronic structure codes, defaults to 'wien2k'
    :type order: str | list[str], optional
    :raises Exception: Throws an error if d-orbital does not match names.
    :return: the valence states corresponding to the d-orbtials
    :rtype: list[YlmExpansion]
    """
    if isinstance(order,  list): return list(map(_str2dorbital, order))
    valid:list[str] = ['wien2k']
    if order not in valid: Exception(f"The argument {order} is not a valid option! The valid options are {valid}")
    match order:
        case "wien2k": return [DZ2, DXY, DX2Y2, DYZ, DXZ]

def build_core_states(edge:str='l2') -> list[YlmExpansion]: # TODO: swithch to enum?
    """The core states for the a given RIXS edge process

    :param edge: The RIXS edge process, defaults to 'l2'
    :type edge: str, optional
    :return: the core states corresponding to the RIXS process given by edge.
    :rtype: list[YlmExpansion]
    """
    match edge:
        case 'l2':
            return [ YlmExpansion(l=1, data= { (+1,0) : 1.0 } ),
                    YlmExpansion(l=1, data= { (+1,+1) : 1.0/np.sqrt(3), (0,0) : np.sqrt(2)/np.sqrt(3) }),
                    YlmExpansion(l=1, data= { (-1,0) : 1.0/np.sqrt(3.), (0,+1): np.sqrt(2)/np.sqrt(3) }),
                    YlmExpansion(l=1, data= { (-1,+1) : 1.0} )
                  ]

# computes the integral ∫ dΩ Y(lc, mc)*Y(l=1,mq)*Y(ld, md), which
# reduces to gaunt coefficients, G(mc, mq, md). 
def _dipole(core_state:YlmExpansion, state:YlmExpansion, polarization:YlmExpansion) -> complex:
    spin_flip = lambda m : -1 if abs(m) == 1 else 1
    result = 0.0+0.0j
    for (m_d, coeff_d) in state:
        for (m_c, spin_c, coeff_c) in core_state:
            for (m_q, coeff_q) in polarization:
                result += spin_flip(m_c)*coeff_c*coeff_d*coeff_q*gaunt(m_c, m_q, m_d)
    return result

# computes the transition amplitude | ∑c <f|ε'|c><c|ε|i>.|^2 
# where ε, ε' are the incoming and outgoing photon polarizations,
# |c> are the core states and |i>, |f> are the initial and 
# final valence states. 
def _initial_to_final_transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          final:YlmExpansion, 
                          incoming_pol:YlmExpansion, 
                          outgoing_pol:YlmExpansion) -> float:
    return abs( sum([np.conj(_dipole(core_state, final, outgoing_pol))*_dipole(core_state, initial, incoming_pol) for core_state in core_states]) )**2

# computes the transition amplitude | ∑c <i|ε|c> |^2 
# where ε is the photon polarization, |c> are the core 
# states and |i> is valence state. 
def _transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          pol:YlmExpansion) -> float:
    return abs( sum(  [_dipole(core_state, initial, pol) for core_state in core_states]) )**2


def rixs_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], 
                         incoming_pol:YlmExpansion, outgoing_pols:list[YlmExpansion]|YlmExpansion ) -> np.ndarray:
    """Compute the RIXS matrix elements at fixed incoming photon polarization and summed over outgoing photon polarizations.

    :param states: a list of valence states.
    :type states: list[YlmExpansion]
    :param core_states: a list of core states.
    :type core_states: list[YlmExpansion]
    :param incoming_pol: incoming photon polarization.
    :type incoming_pol: YlmExpansion
    :param outgoing_pols: a list of outgoing photon polarizations.
    :type outgoing_pols: list[YlmExpansion] | YlmExpansion
    :return: RIXS matrix elements for the initial and final valence states.
    :rtype: np.ndarray
    """
    outgoing_pols = [outgoing_pols] if not isinstance(outgoing_pols, list) else outgoing_pols
    dim = len(states)
    M = np.zeros((dim, dim), dtype=float)
    for pol in outgoing_pols: 
        for initial in range(dim):
            for final in range(dim):
                M[initial, final] += _initial_to_final_transition_amplitude(core_states, states[initial], states[final], incoming_pol, pol)
    return M

def xas_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], polarization:YlmExpansion) -> np.ndarray:
    """Computes the XAS matrix elements for a given polarization.

    :param states: a list of valence states
    :type states: list[YlmExpansion]
    :param core_states: a list of core states
    :type core_states: list[YlmExpansion]
    :param polarization: incoming photon polarizations
    :type polarization: YlmExpansion
    :return: XAS matrix elements
    :rtype: np.ndarray
    """
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

    pol_matrix_elements = pol_matrix_elements if pol_matrix_elements is not None else np.eye(dim_states) # check for correctness

    dim_pol = pol_matrix_elements.shape[0]
    assert dim_states == dim_pol, "The dimension of the density of states is not equal to the number of polarization elements ({dim_states} != {dim_pol})"

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
                       Emax:float=10.0) -> tuple[nd.array, np.ndarray]:
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
    dim_pol = len(pol_matrix_elements)
    assert dim_states == dim_pol, "The dimension of the density of states is not equal to the number of polarization elements ({dim_states} != {dim_pol})"

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
