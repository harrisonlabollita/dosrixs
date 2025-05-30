# Examples using dosrixs

Here we provide a few examples to see how to build a RIXS and/or XAS calculation using dosrixs. The example in ``example1.py`` reproduces the results in figures 4 and 5 of [](). We provide all of the tooling to flexibly describe the geometry, the polarizations, and states that are used in your calculation. Our API coupled with an electronic structure calculation of the density of states allows for virtually any kind of calculation.

```python
# define orbital states
d_orbitals = [DZ2, DXY, DX2Y2, DYZ, DXZ]
core_states = build_core_states()


# define photon polarizations
phi = np.deg2rad(180) 
theta = np.deg2rad(15)
theta_prime = np.deg2rad(15-153)

EX, EY, EZ = build_electric_fields(normal='z');

s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
p_pol      = np.cos(theta)*EZ - np.sin(theta)*(np.cos(phi)*EX + np.sin(phi)*EY)             # p
pprime_pol = np.cos(theta_prime)*EZ - np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'

# compute matrix elements \sum_ϵ' M_if(ϵ,ϵ')
#--> computing rixs for s and p polarizations
Ms_rixs = rixs_matrix_elements(d_orbitals, core_states, s_pol, [s_pol, pprime_pol])
Mp_rixs = rixs_matrix_elements(d_orbitals, core_states, p_pol, [s_pol, pprime_pol])

#--> computing xas for s and p polarizations
Ms_xas = xas_matrix_elements(d_orbitals, core_states, s_pol)
Mp_xas = xas_matrix_elements(d_orbitals, core_states, p_pol)

# compute RISX cross section using DOS and matrix elements
e_mesh, dos = get_density_of_states('data/ndnio2-3d-dos.lda.txt')

#--> computing s pol rixs cross section
s_cross_section = rixs_cross_section(e_mesh, dos, Ms_rixs, Emin = -1.5, Emax = +7.0)
#--> computing p pol rixs cross section
p_cross_section = rixs_cross_section(e_mesh, dos, Mp_rixs, Emin = -1.5, Emax = +7.0)

#--> computing s pol xas
s_xas = xas(e_mesh, dos, Ms_xas, Emin = -1.5, Emax = +7.0)
#--> computing p pol xas
p_xas = xas(e_mesh, dos, Mp_xas, Emin = -1.5, Emax = +7.0)
```

The results of this calculation are shown below:

![](data/example1-output.png)
