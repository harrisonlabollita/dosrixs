import numpy as np
import matplotlib.pyplot as plt; plt.style.use('publish')
import time
from dosrixs import *

def get_density_of_states(filename):
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
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    return ax

def plot_integrated_cross_section(ax, data, emin=4, emax=6, **kwargs):
    ein        = data[0][0,:]
    eloss      = data[1][:,0]
    integrated = data[2][:, np.logical_and(eloss > emin, eloss < emax)].sum(axis=1)
    ax.plot(ein, integrated, **kwargs)
    return ax


# define orbital states
d_orbitals = [DZ2, DXY, DX2Y2, DYZ, DXZ]
core_states = build_l2_core_states()

phi = np.deg2rad(180)
theta = np.deg2rad(15)
theta_prime = np.deg2rad(15-153)

s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
p_pol      = np.cos(theta)*EZ - np.sin(theta)*(np.cos(phi)*EX + np.sin(phi)*EY)             # p
pprime_pol = np.cos(theta_prime)*EZ - np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'


# compute matrix elements \sum_ϵ' M_if(ϵ,ϵ')
print("--> computing s and p polarizations", end= " ")
start = time.perf_counter()
Ms = dipole_matrix_elements(d_orbitals, core_states, s_pol, [s_pol, pprime_pol])
Mp = dipole_matrix_elements(d_orbitals, core_states, p_pol, [s_pol, pprime_pol])
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

fig, ax = plt.subplots(figsize=(3,2))
ax= plot_integrated_cross_section(ax, s_cross_section, lw=1, color='xkcd:blue')
ax= plot_integrated_cross_section(ax, p_cross_section, lw=1, color='xkcd:red')
plt.show()
