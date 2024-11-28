import numpy as np
import matplotlib.pyplot as plt
import h5py
import networkx as nx
import mplhep as hep

import plotting

mcmc_chain = "both_leptonReco_3MeV_woBG_PDGlike_mcmc.jld2"


with h5py.File('outputs/' + mcmc_chain, 'r') as f:
    sin2_th12 = np.array(f['sin2_th12'][()])
    sin2_th13 = np.array(f['sin2_th13'][()])
    dm2_21 = np.array(f['dm2_21'][()])
    stepno = np.array(f['stepno'][()])
    chainID = np.array(f['chainid'][()])

    # Create a boolean mask for stepno > 20000
    mask = stepno > 100 #30000

    # Apply the mask to filter the data
    sin2_th12_burnin = sin2_th12[mask]
    sin2_th13_burnin = sin2_th13[mask]
    dm2_21_burnin = dm2_21[mask]


variables = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\Delta m^2_{21}$']
data = [sin2_th12_burnin, sin2_th13_burnin, dm2_21_burnin]


# Generate weights
# RC = plotting.gaussian(sin2_th13_burnin, 0.022, 0.0007)
fig, axes = plotting.plot_corner(variables, data, externalContours=True, color='#006c94', weights=None)


fig.savefig('images/both_leptonReco_3MeV_woBG_PDGlike.png', dpi=300, format='png')


plt.show()