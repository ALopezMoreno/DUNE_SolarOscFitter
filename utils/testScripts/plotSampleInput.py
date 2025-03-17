import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

plt.rcParams['text.usetex'] = True
hep.style.use("CMS")

enu, ereco = np.genfromtxt("inputs/ES_Enu_Ereco_numutau_250kevts_flatdistrib_noCrossSection.csv", delimiter=',', skip_header=True).T
# enu, ereco = np.genfromtxt("inputs/ES_Enu_Ereco_nue_250kevts_flatdistrib_noCrossSection.csv", delimiter=',', skip_header=True).T
# enu, ereco = np.genfromtxt("inputs/CC_Enu_Ereco_nue_250kevts_flatdistrib_noCrossSection.csv", delimiter=',', skip_header=True).T
# enu, ereco = np.genfromtxt("inputs/CC_Enu_Ereco_nue_250kevts_fluxdistrib.csv", delimiter=',', skip_header=True).T

#ereco = np.genfromtxt("inputs/Ereco_NGamma_1e6-1_evts.csv", delimiter=',', skip_header=True).T * 1e-3

#enu = np.random.uniform(0, 10, 10**7)
nbins = 50

#smear with 5% resolution exponentially
raw_smear = np.random.exponential(0.05, size=len(enu))
smear_clipped = np.clip(raw_smear, 0, 1)

enu_smeared = enu * (1-smear_clipped) * np.random.uniform(0.9,1.1,size=len(enu)) - 2e5*enu/(enu**8+1) * np.random.uniform(0,1,size=len(enu))
random_values = np.random.uniform(0, enu, size=len(enu))
enu_smeared = np.where(enu_smeared < 0, random_values, enu_smeared)
reco_smeared = ereco * np.random.normal(1, 0.1, size=len(enu))


hist_enu, bins_enu = np.histogram(enu, bins=nbins, density=False)
hist_ereco, bins_ereco = np.histogram(ereco, bins=nbins, density=False)
hist_esmeared, bins_esmeared = np.histogram(enu_smeared, bins=nbins, density=False)
hist_recosmeared, bins_recosmeared = np.histogram(reco_smeared, bins=nbins, density=False)

data_to_save = np.column_stack((enu*1e-3, reco_smeared*1e-3))
np.savetxt("inputs/Enu_Ereco_ES_nuother_smeared.csv", data_to_save, delimiter=',', header='Enu,Ereco', comments='')



fig, ax = plt.subplots(figsize = (8, 5))

#hep.histplot(hist_enu, bins=bins_enu, ax=ax, histtype='step', linewidth=2, label="Incoming flux")
#hep.histplot(hist_ereco, bins=bins_ereco, ax=ax, histtype='step', label="Reconstructed flux")
#hep.histplot(hist_esmeared, bins=bins_esmeared, ax=ax, histtype='step', label="Smeared flux")

# Create 2D histogram
hist2d, xedges, yedges = np.histogram2d(enu, reco_smeared, bins=nbins, density=False)

# Normalize each column
hist2d_normalized = hist2d / hist2d.sum(axis=1, keepdims=True)

print(hist2d_normalized)

# Plot the normalized 2D histogram
im1 = ax.imshow(hist2d_normalized.T, origin='lower', aspect='auto',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='viridis', norm=LogNorm())

cbar = plt.colorbar(im1, ax=ax)


ax.set_xlabel(r"$E_\nu$")
ax.set_ylabel(r"$E_{reco}$")
#ax.invert_yaxis()

ax.set_box_aspect(1)
ax.legend()
plt.tight_layout()

plt.savefig('images/response_matrix_ES_nuother_smeared.png', dpi=300)
plt.show()