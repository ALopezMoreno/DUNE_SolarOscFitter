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

dFlux8B_safe = np.where(dFlux8B > 0, dFlux8B, 1e-20)
log_smooth_flux8B = interp1d(energy8B, np.log(dFlux8B_safe), kind='cubic', fill_value="extrapolate")

spectrumHep = model.spectrum("hep")
energyHep = spectrumHep["Energy"].to_numpy()
dFluxHep = spectrumHep["Spectrum"].to_numpy()

dFluxHep_safe = np.where(dFluxHep > 0, dFluxHep, 1e-20)
log_smooth_fluxHep = interp1d(energyHep, np.log(dFluxHep_safe), kind='cubic', fill_value="extrapolate")

def smooth_flux8B(energy):
    return np.exp(log_smooth_flux8B(energy))

def smooth_fluxHep(energy):
    return np.exp(log_smooth_fluxHep(energy))

bin_edges = np.linspace(3, 20, 12)
energies = (bin_edges[:-1] + bin_edges[1:]) / 2

# Normalise the fluxes to turn them into idealised data:
integral_flux8B = np.trapz(smooth_flux8B(energies), energies)
integral_fluxHep = np.trapz(smooth_fluxHep(energies), energies)

total_flux = integral_flux8B + integral_fluxHep

events8B_es = ntotal_es / total_flux * smooth_flux8B(energies)
eventsHep_es = ntotal_es / total_flux * smooth_fluxHep(energies)

events8B_cc = ntotal_cc / total_flux * smooth_flux8B(energies)
eventsHep_cc = ntotal_cc / total_flux * smooth_fluxHep(energies)


fig,ax = plt.subplots(figsize = (8, 5))


#ax.plot(energy8B, dFlux8B, linewidth=2, label=r'$^8$B')
#ax.plot(energyHep, dFluxHep, linewidth=2, label=r'$hep')

ax.plot(energies, events8B_es, color='b', alpha=.7, linewidth=2, label=r'$^8$B es')
ax.plot(energies, eventsHep_es, color='orange', alpha=.7, linewidth=2, label=r'hep es')

ax.plot(energies, events8B_cc, color='b', alpha=.7, linewidth=2, label=r'$^8$B CC', linestyle='dashed')
ax.plot(energies, eventsHep_cc, color='orange', alpha=.7, linewidth=2, label=r'hep CC', linestyle='dashed')

print(energies)
print(events8B_cc)

with h5py.File('inputs/testEvents.jld2', 'w') as f:
    f.create_dataset('energies8B', data=energies*1e-3)
    f.create_dataset('energiesHep', data=energies*1e-3)
    f.create_dataset('events8B_es', data=events8B_es)
    f.create_dataset('events8B_cc', data=events8B_cc)
    f.create_dataset('eventsHep_es', data=eventsHep_es)
    f.create_dataset('eventsHep_cc', data=eventsHep_cc)

#ax.set_yscale('log')

ax.set_xlim(0, 20)
#ax.set_xscale('log')

ax.legend()
plt.savefig('images/agss09_events', dpi=200, format='pdf')

plt.show()