import sys
sys.path.append('../PEANUTS')

import peanuts
import matplotlib.pyplot as plt
import mplhep as hep
import h5py


plt.rcParams['text.usetex'] = True
hep.style.use("CMS")

model = peanuts.SolarModel(solar_model_file="../PEANUTS/Data/nudistr_b16_agss09.dat")

radii = model.radius()
prodFractionBoron = model.fraction("8B")
prodFractionHep = model.fraction("hep")
n_e = model.density()

print(model.fluxes)

exit()
# Save data to a JLD2-compatible HDF5 file
with h5py.File('inputs/AGSS09_high_z.jld2', 'w') as f:
    f.create_dataset('radii', data=radii)
    f.create_dataset('prodFractionBoron', data=prodFractionBoron)
    f.create_dataset('prodFractionHep', data=prodFractionHep)
    f.create_dataset('n_e', data=n_e)


fig, ax = plt.subplots(figsize = (8, 5))
print(radii[-1])
ax.plot(radii, prodFractionBoron, linewidth=2, label=r'$^8$B')
ax.plot(radii, prodFractionHep, linewidth=2, label=r'hep')
ax.legend()
ax.set_xlabel(r'$R/R_\odot$')
ax.set_ylabel(r'Fraction of $\nu_e$ production')



plt.show()