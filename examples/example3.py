import numpy as np
import matplotlib.pyplot as plt;
import time
from dosrixs import *

def get_density_of_states(filename):
    data = np.loadtxt(filename)
    eF = 10.7577
    e = data[:,0]-eF
    dos = 2*data[:,2:]
    return e, dos

def plot_cross_section(ax, data, **kwargs):
    ax.tick_params(which ='both', direction='out', top=True, right=True)
    img = ax.pcolormesh(data[-3], data[-2],data[-1].T, **kwargs)
    cbar = fig.colorbar(img, ax=ax, location='top', orientation='horizontal', shrink=0.7)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    return ax

def plot_integrated_cross_section(ax, data, emin=4, emax=6, **kwargs):
    ax.tick_params(which='both', top=True, right=True)
    ein        = data[0][0,:]
    eloss      = data[1][:,0]
    integrated = data[2][:, np.logical_and(eloss > emin, eloss < emax)].sum(axis=1)
    ax.plot(ein, integrated, **kwargs)
    return ax

def plot_xas(ax, data, **kwargs):
    ax.tick_params(which='both', top=True, right=True)
    ein, xas = data
    ax.plot(ein, xas, **kwargs)
    return ax

# define orbital states
d_orbitals = [DZ2, DXZ, DYZ, DX2Y2, DXY]
core_states = build_l2_core_states()

phi = np.deg2rad(180)
theta = np.deg2rad(15)
theta_prime = np.deg2rad(15-153)

s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
p_pol      = np.cos(theta)*EZ - np.sin(theta)*(np.cos(phi)*EX + np.sin(phi)*EY)             # p
pprime_pol = np.cos(theta_prime)*EZ - np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'

# compute matrix elements \sum_ϵ' M_if(ϵ,ϵ')
print("--> computing rixs for s and p polarizations", end= " ")
start = time.perf_counter()
Ms_rixs = rixs_matrix_elements(d_orbitals, core_states, s_pol, [s_pol, pprime_pol])
Mp_rixs = rixs_matrix_elements(d_orbitals, core_states, p_pol, [s_pol, pprime_pol])
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

print("--> computing xas for s and p polarizations", end= " ")
start = time.perf_counter()
Ms_xas = xas_matrix_elements(d_orbitals, core_states, s_pol)
Mp_xas = xas_matrix_elements(d_orbitals, core_states, p_pol)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

# compute RISX cross section using DOS and matrix elements
e_mesh, dos = get_density_of_states('data/liv2o4-cubic-dft.pdos.pdos_atm#3(V)_wfc#3(d)')

print('--> computing s pol rixs cross section', end = " ")
start = time.perf_counter()
s_cross_section = rixs_cross_section(e_mesh, dos, Ms_rixs, Emin = -1.0, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )
start = time.perf_counter()
print('--> computing p pol rixs cross section', end= " ")
p_cross_section = rixs_cross_section(e_mesh, dos, Mp_rixs, Emin = -1.0, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

print('--> computing s pol xas', end = " ")
start = time.perf_counter()
s_xas = xas(e_mesh, dos, Ms_xas, Emin = -1.0, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )
start = time.perf_counter()
print('--> computing p pol xas', end= " ")
p_xas = xas(e_mesh, dos, Mp_xas, Emin = -1.0, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

from matplotlib.colors import LinearSegmentedColormap
def colormap(*args): return LinearSegmentedColormap.from_list('', list(args), N=256)

# plot results
vmax = np.max(s_cross_section[-1])
fig, ax = plt.subplots(2,2, figsize=(5,7), height_ratios=[2,1])
for a in ax[0,:]:  a.set_xlabel(r"E$_{\text{in}}$ (eV)")
for a  in ax[0,:]: a.set_ylabel(r"E$_{\text{loss}}$ (eV)")
ax[0,0] = plot_cross_section(ax[0,0], s_cross_section,  vmin=0, vmax=vmax, cmap='rainbow'); 
ax[0,0].text(0.6, 0.1, 's pol', transform=ax[0,0].transAxes, color='w')
ax[0,1] = plot_cross_section(ax[0,1], p_cross_section,  vmin=0, vmax=vmax, cmap='rainbow')
ax[0,1].text(0.6, 0.1, 'p pol', transform=ax[0,1].transAxes, color='w')

ax[1,0]= plot_integrated_cross_section(ax[1,0], s_cross_section, lw=1, color='xkcd:blue', label='s')
ax[1,0]= plot_integrated_cross_section(ax[1,0], p_cross_section, lw=1, color='xkcd:red', label='p')
ax[1,0].set_ylabel('Integrated RIXS'); ax[1,0].set_xlabel(r"E$_{\text{in}}$ (eV)"); ax[1,0].legend(loc='best'); ax[1,0].set_ylim(0, )

ax[1,1]= plot_xas(ax[1,1], s_xas, lw=1, color='xkcd:blue', label='s')
ax[1,1]= plot_xas(ax[1,1], p_xas, lw=1, color='xkcd:red', label='p')
ax[1,1].set_xlabel(r"E (eV)"); ax[1,1].set_ylabel("XAS"); ax[1,1].legend(loc='best'); ax[1,1].set_ylim(0, )
plt.subplots_adjust(hspace=0.25, wspace=0.3)
plt.show()
