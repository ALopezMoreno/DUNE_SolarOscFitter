# DEPRECATED: superseded by utils/make_solar_model_inputs.jl (SF-III, SPEC 8.2). This script built
# inputs/AGSS09_high_z.jld2 from B16-AGSS09met, which is the LOW-Z model despite the file name.
# Do not run it: it would overwrite a file the regression fixture uses.
import sys
sys.path.append('../PEANUTS')

import peanuts
import matplotlib.pyplot as plt
import h5py

import style  # applies the shared CMS/LaTeX style on import

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