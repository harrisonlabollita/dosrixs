import time
import numpy as np
import matplotlib.pyplot as plt;
from matplotlib.colors import LinearSegmentedColormap
def colormap(*args): return LinearSegmentedColormap.from_list('', list(args), N=256)

from dosrixs import *

def get_density_of_states(filename):
    data = np.loadtxt(filename)
    eF = 10.7577
    e = data[:,0]-eF
    dos = 2*data[:,1].reshape(-1,1)
    return e, dos

def plot_cross_section(ax, data, **kwargs):
    ax.tick_params(which ='both', direction='out', top=True, right=True)
    img = ax.pcolormesh(data[-3], data[-2],data[-1].T, **kwargs)
    cbar = fig.colorbar(img, ax=ax, location='top', orientation='horizontal', shrink=0.7)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    return ax


# compute RISX cross section using DOS and matrix elements
e_mesh, dos = get_density_of_states('data/lvo-cubic-V3d-pdos.txt')

print('--> computing rixs cross section', end = " ")
start = time.perf_counter()
cross_section = rixs_cross_section(e_mesh, dos, Emin = 0.0, Emax = +6.0)
print("finished in  {:.6f} seconds".format(time.perf_counter()-start) )

# plot results
vmax = np.max(cross_section[-1])
fig, ax = plt.subplots(figsize=(2.5,4))
ax.set_xlabel(r"E$_{\text{in}}$ (eV)")
ax = plot_cross_section(ax, cross_section,  vmin=0, vmax=vmax, cmap='rainbow')
ax.set_ylabel(r"E$_{\text{loss}}$ (eV)")
plt.show()
