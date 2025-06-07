import unittest
import numpy as np
from dosrixs import build_core_states, build_d_states, build_electric_fields
from dosrixs import rixs_matrix_elements
from dosrixs.ylmexpansion import YlmExpansion
from dosrixs.utils import gaunt

g = lambda x, y, z: gaunt(m1=x,m2=y,m3=z)

def _generate_reference_matrix_elements(theta, thetap):

    cs, sn = np.cos(theta), np.sin(theta)
    csp, snp = np.cos(thetap), np.sin(thetap)
    s3, s2 = 1/np.sqrt(3), 1/np.sqrt(2)
    s23 = s3/s2

    ac = np.zeros((4,3,2), dtype=float)
    ac[0, 2, 0] = 1.0; ac[1, 1, 0] = s23; ac[2, 1, 1] = s23
    ac[1, 2, 1] = s3;  ac[2, 0, 0] = s3;  ac[3, 0, 1] = 1.0

    b = np.zeros((5,5), dtype=float)
    b[0,0] = -s2; b[1,1] = -s2;  b[3,1] = s2; b[4,0] = s2
    b[0,4] =  s2; b[1,3] =  s2;  b[3,3] = s2; b[4,4] = s2
    b[2,2] = 1.0


    cf = np.zeros((2,5,5)) 

    for ii in range(5):
        for ifd in range(5):
            zss, zsp, zps, zpp = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))
            for iss in range(2):
                for isp in range(2):
                    for j in range(4):
                        facsp, facpp = 0, 0
                        for im, m in enumerate([-2,-1,0,1,2]):
                            im_sign = -1 if abs(m) == 1 else 1
                            for inn, n in enumerate([-1,0,1]):
                                facsp += (s2 * (g(n,1,-m) + g(n,-1,-m))* im_sign) * b[ifd, im] * ac[j,inn,isp]
                                facpp += ((snp*s2*(-g(n,1,-m)+g(n,-1,-m)) + csp*g(n,0,-m)) * im_sign) *b[ifd,im]*ac[j,inn,isp]
                        facs, facp = 0, 0
                        for il, l in enumerate([-2,-1,0,1,2]):
                            for inn, n in enumerate([-1,0,1]):
                                in_sign = -1 if n !=  0 else 1
                                facs += (s2 * (g(-n,1,l) + g(-n, -1, l))* in_sign) * b[ii, il]*ac[j,inn,iss]
                                facp += ((sn*s2*(-g(-n,1,l) + g(-n,-1,l)) + cs *g(-n,0,l))*in_sign) * b[ii,il] * ac[j,inn,iss]
                        zss[iss,isp] += facs * facsp
                        zsp[iss,isp] += facs * facpp
                        zps[iss,isp] += facp * facsp
                        zpp[iss,isp] += facp * facpp
            cf[0,ii,ifd] = np.sum(zss**2) + np.sum(zsp**2)
            cf[1,ii,ifd] = np.sum(zps**2) + np.sum(zpp**2)
    return cf


def _calculate_matrix_elements(incoming:YlmExpansion, outgoing:list[YlmExpansion]) -> np.ndarray:
    return rixs_matrix_elements(d_orbitals, core_states, incoming, outgoing)

# define orbitals and core states
EX, EY, EZ = build_electric_fields(normal='z')
d_orbitals = build_d_states(order=['dxy','dxz', 'dz2', 'dyz', 'dx2y2'])
core_states = build_core_states('L3')
phi = np.deg2rad(180)
test_angles:list[tuple[int,int]] = [(x,y) for (x,y) in zip([10,20,45,50,90], [90,50,45,20,10])]

class TestMatrixElements(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref_pol_rixs_data = { (theta, thetap) : _generate_reference_matrix_elements(np.deg2rad(theta), 
                                                                                        np.deg2rad(thetap)) for (theta, thetap) in test_angles }
    def test_rixs_spol_matrix_elements(self):
        for (th, thp) in test_angles:
            with self.subTest(th=th, thp=thp):
                theta, theta_prime = np.deg2rad(th), np.deg2rad(thp)
                s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
                pprime_pol = np.cos(theta_prime)*EZ + np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'
                pol_rixs     = _calculate_matrix_elements(s_pol, [s_pol, pprime_pol])
                np.testing.assert_allclose(self.ref_pol_rixs_data[(th, thp)][0], pol_rixs[0], rtol=1e-12, atol=1e-12)

    def test_rixs_ppol_matrix_elements(self):
        for (th, thp) in test_angles:
            with self.subTest(th=th, thp=thp):
                theta, theta_prime = np.deg2rad(th), np.deg2rad(thp)
                s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
                p_pol      = np.cos(theta)*EZ + np.sin(theta)*(np.cos(phi)*EX + np.sin(phi)*EY)             # p
                pprime_pol = np.cos(theta_prime)*EZ + np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'
                pol_rixs     = _calculate_matrix_elements(p_pol, [s_pol, pprime_pol])
                np.testing.assert_allclose(self.ref_pol_rixs_data[(th,thp)][1], pol_rixs[0], rtol=1e-12, atol=1e-12)