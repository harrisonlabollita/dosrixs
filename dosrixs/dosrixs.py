from typing import Literal
from itertools import product

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

    Returns (EX, EY, EZ) — the three Cartesian polarization directions
    expressed as l=1 spherical harmonic expansions. The labeling is always
    in the lab frame; the ``normal`` parameter selects which axis is the
    surface normal for convenience when constructing s/p polarizations.

    :param normal: The surface normal direction ('x', 'y', or 'z'), defaults to 'z'
    :type normal: str, optional
    :raises ValueError: If normal is not 'x', 'y', or 'z'
    :return: The electric fields (EX, EY, EZ).
    :rtype: tuple[YlmExpansion, YlmExpansion, YlmExpansion]
    """
    s2 = 1.0 / np.sqrt(2)
    EX = YlmExpansion(l=1, data={(-1, 0):  s2, (0, 0): 0.0, (+1, 0): -s2})
    EY = YlmExpansion(l=1, data={(-1, 0): 1j*s2, (0, 0): 0.0, (+1, 0): 1j*s2})
    EZ = YlmExpansion(l=1, data={(-1, 0): 0.0, (0, 0): 1.0, (+1, 0): 0.0})
    if normal == 'z':
        return EX, EY, EZ
    elif normal == 'x':
        return EY, EZ, EX
    elif normal == 'y':
        return EZ, EX, EY
    else:
        raise ValueError(f"The surface normal = '{normal}' is not a valid option. Use 'x', 'y', or 'z'.")

DORBITAL = Literal["dxy", "dx2y2", "dz2", "dxz", "dyz"]

EDGE = Literal["L2", "L3"]

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

def build_local_d_states(U:np.ndarray|None=None, order:list[DORBITAL]=["dz2", "dx2y2", "dxy", "dxz", "dyz"]) -> list[YlmExpansion]:
    r"""Build valence states from a local-to-cubic-harmonic rotation matrix U.

    Given a unitary matrix U that relates a local d-orbital basis to the
    standard cubic harmonics, this function constructs the local orbitals
    as :class:`YlmExpansion` objects suitable for matrix element calculations.

    The local orbitals are defined as:

    .. math:: |\tilde{d}_\alpha\rangle = \sum_\beta U_{\beta\alpha} |d_\beta\rangle

    where :math:`|d_\beta\rangle` are the standard cubic harmonics in the given order.

    :param U: unitary rotation matrix (5×5) from local to cubic harmonic basis,
              where column α gives the cubic harmonic coefficients of local orbital α.
              Defaults to the identity matrix (standard cubic harmonics).
    :type U: np.ndarray | None
    :param order: ordering of the cubic harmonic d-orbitals matching U's rows
    :type order: list[DORBITAL]
    :return: local d-orbital states as YlmExpansion objects
    :rtype: list[YlmExpansion]
    """
    if U is None:
        U = np.eye(5)
    cubic_states = build_d_states(order)
    local_states = []
    for alpha in range(U.shape[1]):
        state = U[0, alpha] * cubic_states[0]
        for beta in range(1, len(cubic_states)):
            state = state + U[beta, alpha] * cubic_states[beta]
        local_states.append(state)
    return local_states

def cubic_to_spherical_matrix(order:list[DORBITAL]=["dz2", "dx2y2", "dxy", "dxz", "dyz"]) -> np.ndarray:
    r"""Build the unitary transformation matrix T from the cubic harmonic
    (real d-orbital) basis to the complex spherical harmonic basis.

    The rows are indexed by :math:`m = -2, -1, 0, 1, 2` and the columns
    by the cubic harmonic orbitals in the given order.

    To transform a matrix U from the cubic to the spherical harmonic basis:

    .. math:: U^{Y_{lm}} = T \, U \, T^{\dagger}

    :param order: ordering of the cubic harmonic d-orbitals
    :type order: list[DORBITAL]
    :return: the 5×5 unitary transformation matrix T
    :rtype: np.ndarray
    """
    d_states = build_d_states(order)
    T = np.zeros((5, 5), dtype=complex)
    for col, state in enumerate(d_states):
        for m in range(-2, 3):
            T[m + 2, col] = state[(m, 0)]
    return T

def rotate_matrix_to_spherical(U:np.ndarray, order:list[DORBITAL]=["dz2", "dx2y2", "dxy", "dxz", "dyz"]) -> np.ndarray:
    r"""Rotate a matrix from the cubic harmonic basis to the spherical harmonic basis.

    .. math:: U^{Y_{lm}} = T \, U \, T^{\dagger}

    :param U: a matrix defined in the cubic harmonic d-orbital basis
    :type U: np.ndarray
    :param order: ordering of the cubic harmonic d-orbitals matching the rows/columns of U
    :type order: list[DORBITAL]
    :return: the matrix in the spherical harmonic (m = -2, -1, 0, 1, 2) basis
    :rtype: np.ndarray
    """
    T = cubic_to_spherical_matrix(order)
    return T @ U @ T.conj().T

def get_core_edge_quantum_numbers(edge:EDGE) -> dict[str, float]:
    """Get the quantum number corresponding to an edge.

    :param edge: Core edge
    :type edge: EDGE
    :return: quantum numbers for edge
    :rtype: dict[str, float]
    """
    if   edge == "L2":  return {'n' : 2, 'l' : 1, 'j' : 0.5}
    elif edge == "L3":  return {'n' : 2, 'l' : 1, 'j' : 1.5}
    else: raise ValueError(f"Unsupported edge: '{edge}'. Supported edges: L2, L3.")

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
    loop = int(j*2)
    core_states:list[YlmExpansion] = []
    for tmj in np.arange(-loop, loop+1, 2):
        mj= float(0.5*tmj)
        data = {}
        for m in range(-l, l+1):
            for spin in [-s, s]:
                data[(m, spin2idx(spin))] = gaunt_sympy(l, m, s, spin, j, mj)
        core_states.append(YlmExpansion(l=l, data=data))
    return core_states

def initial_to_final_transition_amplitude(core_states:list[YlmExpansion], 
                          initial:YlmExpansion, 
                          final:YlmExpansion, 
                          incoming_pol:YlmExpansion, 
                          outgoing_pol:YlmExpansion) -> float:
    r"""
    Computes the transition amplitude:
    
    .. math:: \sum_{c} \langle f | \epsilon' | c \rangle \langle c | \epsilon | i \rangle,

    where ε, ε' are the incoming and outgoing photon polarizations,
    |c> are the core states and |i>, |f> are the initial and 
    final valence states. 
    """
    lc = core_states[0].angular_quantum_number
    lp_in = incoming_pol.angular_quantum_number
    lp_out = outgoing_pol.angular_quantum_number
    li = initial.angular_quantum_number
    lf = final.angular_quantum_number

    # Pre-collect non-zero (m, spin, coeff) entries to avoid redundant lookups
    initial_terms = [(m, s, c) for m, s, c in initial if abs(c) > 0]
    final_terms   = [(m, s, c) for m, s, c in final   if abs(c) > 0]
    pol_in_terms  = [(mq, sq, cq) for mq, sq, cq in incoming_pol  if abs(cq) > 0]
    pol_out_terms = [(mq, sq, cq) for mq, sq, cq in outgoing_pol  if abs(cq) > 0]

    total = np.zeros((2, 2), dtype=complex)
    for core in core_states:
        core_terms = [(mc, sc, cc) for mc, sc, cc in core if abs(cc) > 0]
        for sp1, sp2 in product(range(2), range(2)):
            dipole_out = 0.0j
            for m, s, cf in final_terms:
                if s != sp2: continue
                for mc, sc, cc in core_terms:
                    if sc != sp2: continue
                    for mq, _, cq in pol_out_terms:
                        if mc + mq + m != 0: continue
                        dipole_out += cf * cc * cq * gaunt(lc, mc, lp_out, mq, lf, m)

            dipole_in = 0.0j
            for m, s, ci in initial_terms:
                if s != sp1: continue
                for mc, sc, cc in core_terms:
                    if sc != sp1: continue
                    for mq, _, cq in pol_in_terms:
                        if mc + mq + m != 0: continue
                        dipole_in += ci * cc * cq * gaunt(lc, mc, lp_in, mq, li, m)

            total[sp1, sp2] += np.conj(dipole_out) * dipole_in
    return float(np.sum(np.abs(total)**2))

def rixs_matrix_elements(states:list[YlmExpansion], core_states:list[YlmExpansion], 
                         incoming_pols:list[YlmExpansion]|YlmExpansion, outgoing_pols:list[YlmExpansion]|YlmExpansion ) -> np.ndarray:
    r"""Compute the RIXS matrix elements at fixed incoming photon polarization and summed over outgoing photon polarizations:

    .. math:: M_{\epsilon, i, f} = \sum_{\epsilon', c, \sigma, \sigma'} \Big | \langle f \sigma' | \epsilon' | c \rangle \langle c | \epsilon | i \sigma \rangle \Big |^{2} 


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
    for initial, final in product(range(dim_states), range(dim_states)):
        for (in_pol, incoming) in enumerate(incoming_pols):
            for outgoing in outgoing_pols: 
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
            M[ip, initial] = initial_to_final_transition_amplitude(core_states, states[initial], states[initial],pol, pol)
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

    dim_pol_mat = pol_matrix_elements.shape[1]
    if dim_states != dim_pol_mat:
        raise Exception(f"The dimension of the density of states is not equal to the number of polarization elements ({dim_states} != {dim_pol_mat})")

    below_zero, above_zero = np.where(e_mesh < 0.0)[0], np.where(e_mesh > 0.0)[0]

    eocc, rho_occ = e_mesh[below_zero], density_of_states[below_zero, :]

    Ein, Eloss = e_mesh[(e_mesh > Emin) & (e_mesh < Emax)], e_mesh[(e_mesh > 0.0) & (e_mesh < Emax)]

    x_grid, y_grid = np.meshgrid(Ein, Eloss)

    e_interp_all, eout_all = eocc[:, None] + Eloss[None, :], Ein[:, None] - Eloss[None, :]

    lorentz = 0.5 * Gamma / ((eocc[:,None, None]-eout_all[None, :, :])**2 + 0.25*Gamma**2)

    rho_shifted_all = np.empty((dim_states, len(eocc), len(Eloss)))
    for i in range(dim_states): rho_shifted_all[i,:,:] = np.interp(e_interp_all.flatten(), e_mesh[above_zero], 
                                           density_of_states[above_zero,i], left=0.0, right=0.0
                                           ).reshape(len(eocc), len(Eloss))

    # Contract polarization matrix elements with occupied DOS up front
    # rho_occ: (n_occ, n_states), pol: (n_pol, n_states, n_states)
    # rho_pol: (n_pol, n_occ, n_states) = sum_f rho_occ(e,f) * M(pol,f,i) 
    #   → weight for each (pol, occ_energy, initial_state)
    rho_pol = np.einsum('ef,pfi->pei', rho_occ, pol_matrix_elements)

    # rho_shifted_all: (n_states, n_occ, n_eloss)
    # rho_pol:         (n_pol, n_occ, n_states)
    # lorentz:         (n_occ, n_ein, n_eloss)
    # integrand:       (n_pol, n_occ, n_ein, n_eloss)
    integrand = np.einsum('pei,iel,ekl->pekl', rho_pol, rho_shifted_all, lorentz)

    cross_section = simpson(integrand, x=eocc, axis=1)
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
    assert dim_states == dim_pol_el, f"The dimension of the density of states is not equal to the number of polarization elements ({dim_states} != {dim_pol_el})"

    dim_pol = pol_matrix_elements.shape[0]

    above_zero = np.where(e_mesh > 0.0)[0]

    eunocc = e_mesh[above_zero]
    rho_unocc = density_of_states[above_zero, :]

    Ein   = e_mesh[(e_mesh > Emin) & (e_mesh < Emax)]

    # lorentz: shape (n_unocc, n_ein)
    lorentz = 0.5 * Gamma / ((eunocc[:, None] - Ein[None, :]) ** 2 + 0.25 * Gamma ** 2)

    # rho_weighted: shape (n_pol, n_states, n_unocc) — DOS weighted by matrix elements
    rho_weighted = pol_matrix_elements[:, :, None] * rho_unocc[None, :, :].transpose(0, 2, 1)

    # integrand: shape (n_pol, n_unocc, n_ein) — sum over states, then integrate over unocc
    integrand = np.einsum('pse,ek->pek', rho_weighted, lorentz)
    xas_data = simpson(integrand, x=eunocc, axis=1)

    return Ein, xas_data