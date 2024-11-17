import numpy as np
import matplotlib.pyplot as plt
import h5py
import networkx as nx
import mplhep as hep

import plotting

mcmc_chain = "testFit.jld2"


with h5py.File('outputs/' + mcmc_chain, 'r') as f:
    sin2_th12 = np.array(f['sin2_th12'][()])
    sin2_th13 = np.array(f['sin2_th13'][()])
    dm2_21 = np.array(f['dm2_21'][()])
    stepno = np.array(f['stepno'][()])
    chainID = np.array(f['chainid'][()])

variables = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\Delta m^2_{21}$']
data = data = [sin2_th12, sin2_th13, dm2_21]

fig, axes = plotting.plot_corner(variables, data)
fig.savefig('images/corner.png', dpi=300, format='png')