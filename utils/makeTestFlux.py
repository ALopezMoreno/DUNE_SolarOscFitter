import sys
sys.path.append('../PEANUTS')

import numpy as np
import peanuts
import matplotlib.pyplot as plt
import mplhep as hep
import h5py
from scipy.interpolate import interp1d

plt.rcParams['text.usetex'] = True
hep.style.use("CMS")

ntotal_es = 70 *1e3
ntotal_cc = 125 *1e3

model = peanuts.SolarModel(solar_model_file="../PEANUTS/Data/nudistr_b16_agss09.dat")

spectrum8B = model.spectrum("8B")
energy8B = spectrum8B["Energy"].to_numpy()
dFlux8B = spectrum8B["Spectrum"].to_numpy()
totalFlux8B = model.flux("8B")

dFlux8B_safe = np.where(dFlux8B > 0, dFlux8B, 1e-20)
log_smooth_flux8B = interp1d(energy8B, dFlux8B_safe, kind='cubic', fill_value="extrapolate")

spectrumHep = model.spectrum("hep")
energyHep = spectrumHep["Energy"].to_numpy()
dFluxHep = spectrumHep["Spectrum"].to_numpy()
totalFluxHep = model.flux("hep")


dFluxHep_safe = np.where(dFluxHep > 0, dFluxHep, 1e-20)
log_smooth_fluxHep = interp1d(energyHep, dFluxHep_safe, kind='cubic', fill_value="extrapolate")

def smooth_flux8B(energy):
    return (log_smooth_flux8B(energy))

def smooth_fluxHep(energy):
    return (log_smooth_fluxHep(energy))

bin_edges = np.linspace(0, 20, 1000)
energies = (bin_edges[:-1] + bin_edges[1:]) / 2

# Normalise the fluxes to turn them into idealised data:
integral_flux8B = np.trapz(smooth_flux8B(energies), energies)

integral_fluxHep = np.trapz(smooth_fluxHep(energies), energies)

total_flux = integral_flux8B + integral_fluxHep

flux8B_raw =  np.where(smooth_flux8B(energies) < 0, 0, smooth_flux8B(energies)) 
fluxHep_raw =  np.where(smooth_fluxHep(energies) < 0, 0, smooth_fluxHep(energies))

flux8B = flux8B_raw / np.sum(flux8B_raw) * totalFlux8B
fluxHep = fluxHep_raw / np.sum(fluxHep_raw) * totalFluxHep

print('total sum is ' + str(np.sum(flux8B)))
print('total flux8B is ' + str(totalFlux8B))
print('integral 8B is ' + str(integral_flux8B))
fig,ax = plt.subplots(figsize = (8, 5))


#ax.plot(energy8B, dFlux8B, linewidth=2, label=r'$^8$B')
#ax.plot(energyHep, dFluxHep, linewidth=2, label=r'$hep')

ax.plot(energies, flux8B, color='b', alpha=.7, linewidth=2, label=r'$^8$B es')
ax.plot(energies, fluxHep, color='orange', alpha=.7, linewidth=2, label=r'hep es')


with h5py.File('inputs/fluxes.jld2', 'w') as f:
    f.create_dataset('energies', data=energies*1e-3)
    f.create_dataset('flux8B', data=flux8B)
    f.create_dataset('fluxHep', data=fluxHep)
    f.create_dataset('total8B', data=totalFlux8B)
    f.create_dataset('totalHep', data=totalFluxHep)


#ax.set_yscale('log')

ax.set_xlim(0, 20)


ax.legend()
plt.savefig('images/agss09_events', dpi=200, format='pdf')

plt.show()