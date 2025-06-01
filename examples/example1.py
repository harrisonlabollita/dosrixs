import numpy as np
import matplotlib.pyplot as plt;
import time
from dosrixs import build_core_states, build_d_states, build_electric_fields
from dosrixs import rixs_cross_section, rixs_matrix_elements, xas, xas_matrix_elements

def get_density_of_states(filename):
    data = np.loadtxt(filename)
    e = data[:,0]
    dos = data[:,1:]
    dos[:,-1] /= 2.0
    dos[:,-2] /= 2.0
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
    ax.set_xlim(min(ein), max(ein))
    return ax

def plot_xas(ax, data, **kwargs):
    ax.tick_params(which='both', top=True, right=True)
    ein, xas = data
    ax.plot(ein, xas, **kwargs)
    ax.set_xlim(min(ein), max(ein))
    return ax

# define orbital states
d_orbitals = build_d_states()
core_states = build_core_states('L3')

phi = np.deg2rad(180)
theta = np.deg2rad(15)
theta_prime = np.deg2rad(15-153)

EX, EY, EZ = build_electric_fields(normal='z')

s_pol      = np.sin(phi)*EX - np.cos(phi)*EY                                                # s, s'
p_pol      = np.cos(theta)*EZ + np.sin(theta)*(np.cos(phi)*EX + np.sin(phi)*EY)             # p
pprime_pol = np.cos(theta_prime)*EZ + np.sin(theta_prime)*(np.cos(phi)*EX + np.sin(phi)*EY) # p'

# compute matrix elements \sum_ϵ' M_if(ϵ,ϵ')
print("--> computing rixs for s and p polarizations", end= " ")
start = time.perf_counter()
pol_rixs = rixs_matrix_elements(d_orbitals, core_states, [s_pol, p_pol], [s_pol, pprime_pol])
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

print("--> computing xas for s and p polarizations", end= " ")
start = time.perf_counter()
pol_xas = xas_matrix_elements(d_orbitals, core_states, [s_pol, p_pol])
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

# compute RISX cross section using DOS and matrix elements
e_mesh, dos = get_density_of_states('data/ndnio2-3d-dos.lda.txt')

print('--> computing s pol and p pol rixs cross section', end = " ")
start = time.perf_counter()
rixs = rixs_cross_section(e_mesh, dos, pol_rixs, Emin = -1.5, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

print('--> computing s pol and p pol xas', end = " ")
start = time.perf_counter()
xray = xas(e_mesh, dos, pol_xas, Emin = -1.5, Emax = +7.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

from matplotlib.colors import LinearSegmentedColormap
def colormap(*args): return LinearSegmentedColormap.from_list('', list(args), N=256)

# plot results
vmax = np.max(rixs[-1][0])
fig, ax = plt.subplots(2,2, figsize=(5,7), height_ratios=[2,1])
for a  in ax[0,:]:  a.set_xlabel(r"E$_{\text{in}}$ (eV)")
for a  in ax[0,:]: a.set_ylabel(r"E$_{\text{loss}}$ (eV)")
ax[0,0] = plot_cross_section(ax[0,0], [rixs[0], rixs[1], rixs[2][0]],  vmin=0, vmax=vmax, cmap='rainbow'); 
ax[0,0].text(0.6, 0.1, 's pol', transform=ax[0,0].transAxes, color='w')
ax[0,1] = plot_cross_section(ax[0,1], [rixs[0], rixs[1], rixs[2][1]],  vmin=0, vmax=vmax, cmap='rainbow')
ax[0,1].text(0.6, 0.1, 'p pol', transform=ax[0,1].transAxes, color='w')

ax[1,0]= plot_integrated_cross_section(ax[1,0], [rixs[0], rixs[1], rixs[2][0]], lw=1, color='xkcd:blue', label='s')
ax[1,0]= plot_integrated_cross_section(ax[1,0], [rixs[0], rixs[1], rixs[2][1]], lw=1, color='xkcd:red', label='p')
ax[1,0].set_ylabel('Integrated RIXS'); ax[1,0].set_xlabel(r"E$_{\text{in}}$ (eV)"); ax[1,0].legend(loc='best'); ax[1,0].set_ylim(0, )

ax[1,1]= plot_xas(ax[1,1], [xray[0], xray[1][0]], lw=1, color='xkcd:blue', label='s')
ax[1,1]= plot_xas(ax[1,1], [xray[0], xray[1][1]], lw=1, color='xkcd:red', label='p')
ax[1,1].set_xlabel(r"E (eV)"); ax[1,1].set_ylabel("XAS"); ax[1,1].legend(loc='best'); ax[1,1].set_ylim(0, )
plt.subplots_adjust(hspace=0.25, wspace=0.5)
plt.savefig('data/example1-output.png', bbox_inches='tight')
plt.show()
