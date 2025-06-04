import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from cmap import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
import plotting
import os

plt.rcParams['text.usetex'] = True
hep.style.use("CMS")


G_f = 5.4489e-5


def plot_95_percent_contour(ax, x_data, y_data, color, alpha=1, linestyles=['-'], fill=True):
    # Optionally use a subset of the data for large datasets
    #if len(x_data) > 10000:  # Arbitrary threshold for large datasets
    #    indices = np.random.choice(len(x_data), int(len(x_data) * 0.5), replace=False)
    #    x_data = x_data[indices]
    #    y_data = y_data[indices]

    # Perform kernel density estimation
    xy = np.vstack([x_data, y_data])
    kde = gaussian_kde(xy)
    
    # Create a grid over the data range
    x_min, x_max = x_data.min()*0.9, x_data.max()*1.1
    y_min, y_max = y_data.min()*0.9, y_data.max()*1.1
    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = kde(positions).reshape(x_grid.shape)
    
    # Find the contour level that encloses 99% of the data
    sorted_density = np.sort(density.ravel())
    cumulative_density = np.cumsum(sorted_density)
    cumulative_density /= cumulative_density[-1]
    level_99 = sorted_density[np.searchsorted(cumulative_density, 0.05)]
    
    # Plot the contour
    if fill:
        ax.contour(x_grid * 1e3, y_grid, density, levels=[level_99], colors=color, linewidths=1.5, linestyles=linestyles)
        ax.contourf(x_grid * 1e3, y_grid, density, levels=[level_99, density.max()], colors=[color], alpha=alpha)

    else:
        ax.contour(x_grid * 1e3, y_grid, density, levels=[level_99], colors=color, linewidths=2.5, linestyles=linestyles)


def msw_prob(oscpars, E_true, n_e):
    global G_f
    """
    with h5py.File('outputs/' + mcmc_chain, 'r') as f:
    sin2_th12 = np.array(f['sin2_th12'][()])
    sin2_th13 = np.array(f['sin2_th13'][()])
    dm2_21 = np.array(f['dm2_21'][()])Parameters:
    - oscpars: A dictionary containing oscillation parameters with keys 'sin2_th12', 'sin2_th13', 'dm2_21'.
    - E_true: The true energy value, which must be positive.
    - n_e: The electron density, which must be non-negative.
    """

    
    # Calculate the cosine of twice the angle theta_12 using the oscillation parameter
    c2th12 = np.cos(2 * np.arcsin(np.sqrt(oscpars['sin2_th12'])))
    
    # Calculate the beta parameter
    Acc = 2 * np.sqrt(2) * n_e * G_f * E_true * (1 - oscpars['sin2_th13'])
    beta = Acc / oscpars['dm2_21']
    
    # Calculate the modified cosine of twice the angle theta_12 in matter
    c2th12m = (c2th12 - beta) / np.sqrt((c2th12 - beta)**2 + (1 - c2th12**2))
    
    # Calculate the modified sine squared of theta_13 in matter
    s13m = np.sqrt(oscpars['sin2_th13'])
    
    # Calculate the probability for the electron neutrino flavor
    Probs = 0.5 * (1 - oscpars['sin2_th13']) * (1 - s13m**2) * (1 + c2th12 * c2th12m) + oscpars['sin2_th13'] * s13m**2
    return Probs

def solar_surface_probs(params, E_true, solar_model, process="8B"):
    """
    Calculate the neutrino oscillation probabilities at the solar surface for a given energy and solar model.

    Parameters:
    - params: An instance of OscillationParameters containing the oscillation parameters.
    - E_true: The true energy of the neutrino.
    - solar_model: The solar model object containing radius, electron density, and production regions.
    - process: The process type, either "8B" or "hep".

    Returns:
    - A float representing the integrated probability for the specified neutrino type.
    """

    # Calculate the neutrino oscillation probabilities using the electron density
    enu_osc_prob = msw_prob(params, E_true, solar_model['n_e'])

    # Define a local function to integrate probabilities over the solar radius
    def integrate_prob(radii, prod_fraction, enu_osc_prob):
        # Calculate the bin widths for integration
        b_edges = np.concatenate(([0.0], radii))
        bin_widths = np.diff(b_edges)
        
        # Integrate the probabilities for Boron and Hep neutrinos
        integral = np.sum(prod_fraction * bin_widths * enu_osc_prob)
        
        return integral

    if process == "8B":
        prod_fraction = solar_model['prodFractionBoron']
    elif process == "hep":
        prod_fraction = solar_model['prodFractionHep']
    else:
        raise ValueError("Invalid process specified. Please use '8B' or 'hep'.")

    # Call the integration function with the loaded data
    prob_nue = integrate_prob(solar_model['radii'], prod_fraction, enu_osc_prob)

    return prob_nue


solar_model_file = "inputs/AGSS09_high_z.jld2"
with h5py.File(solar_model_file, "r") as file:
    # Load the necessary datasets from the file
    radii = file["radii"][:]
    prod_fraction_boron = file["prodFractionBoron"][:]
    prod_fraction_hep = file["prodFractionHep"][:]
    n_e = file["n_e"][:]

        # Create a dictionary to store the datasets
    solar_model = {
        "radii": radii,
        "prodFractionBoron": prod_fraction_boron,
        "prodFractionHep": prod_fraction_hep,
        "n_e": n_e
    }



# Open the JLD2 file
file_path = "outputs/unoscillatedSamples.jld2"

with h5py.File(file_path, 'r') as file:
    # Access the datasets
    ES_nue_8B = file['unoscillated_ES_nue_sample_8B'][:]
    ES_nuother_8B = file['unoscillated_ES_nuother_sample_8B'][:]
    CC_8B = file['unoscillated_CC_sample_8B'][:]
    ES_nue_hep = file['unoscillated_ES_nue_sample_hep'][:]
    ES_nuother_hep = file['unoscillated_ES_nuother_sample_hep'][:]
    CC_hep = file['unoscillated_CC_sample_hep'][:]
    energies = file['energies_GeV'][:]*1e3



# Define the range for dm2_21
dm2_21_values = np.asarray([1.5e-4]) #np.linspace(4.95e-5, 2.5e-4, 2)
print(dm2_21_values)
print(dm2_21_values/5 * 1e4-0.09)

# Calculate solar surface probabilities for each dm2_21 and each energy
calc_energies = np.linspace(0, energies[-1], 300) *1e-3



# Create subplots with shared x-axis
fig, (axProbs, ax) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2]})


# Calculate solar surface probabilities for each dm2_21 and each energy
solar_probs = np.array([
    [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': dm2/5 * 1e4-0.09, 'dm2_21': dm2}, energy, solar_model)
     for energy in calc_energies]
    for dm2 in dm2_21_values
])


for probs in solar_probs:
    axProbs.plot(calc_energies*1e3, probs, linestyle=':', color='b', linewidth=2)

solar_probs = np.array([
    [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': dm2/5 * 1e4-0.09, 'dm2_21': dm2}, energy, solar_model, process='hep')
     for energy in calc_energies]
    for dm2 in dm2_21_values
])

for probs in solar_probs:
    axProbs.plot(calc_energies*1e3, probs, color='r', linewidth=2, linestyle=':')

solar_probs8B = np.array(
    [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, energy, solar_model, process='8B')
     for energy in calc_energies])

solar_probshep = np.array(
    [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, energy, solar_model, process='hep')
     for energy in calc_energies])

axProbs.plot(calc_energies*1e3, solar_probs8B, color='b', linewidth=2.5)
axProbs.plot(calc_energies*1e3, solar_probshep, color='r', linewidth=2.5)


axProbs.set_ylim(0, 0.75)

# Plot the first three datasets on the primary y-axis
ax.step(energies, CC_8B, where='mid', label='CC_8B', linestyle='-', color='b', linewidth=2)
ax.step(energies, ES_nue_8B, where='mid', label='ES_nue_8B', linestyle='-.', color='teal', linewidth=2)
ax.step(energies, ES_nuother_8B, where='mid', label='ES_nuother_8B', linestyle=':', color='turquoise', linewidth=2)


# Create a secondary y-axis
ax2 = ax.twinx()

# Plot the last three datasets on the secondary y-axis
ax2.step(energies, CC_hep, where='mid', label='CC_hep', linestyle='-', color='red', linewidth=2)
ax2.step(energies, ES_nue_hep, where='mid', label='ES_nue_hep', linestyle='-.', color='chocolate', linewidth=2)
ax2.step(energies, ES_nuother_hep, where='mid', label='ES_nuother_hep', linestyle=':', color='orange', linewidth=2)

# Set y-axis limits for ax to start from zero
ax.set_ylim(0, ax.get_ylim()[1])

# Set y-axis limits for ax2 to start from zero and be twice as large
ax2.set_ylim(0, ax2.get_ylim()[1] * 2)

# Set x-axis limits
ax.set_xlim([energies.min(), energies.max()])

# Add labels and legend
ax.set_xlabel(r'$E_\nu$ (MeV)')
ax.set_ylabel(r'Oscillated events')
axProbs.set_ylabel(r'$P_{ee}$')

# Position legends outside the plot
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()

# Show the plot
plt.savefig('images/oscillated_sample_etrue.png', dpi=300)

# Load MC
enu_ES_nuother, ereco_ES_nuother = np.genfromtxt("inputs/Enu_Ereco_ES_nuother_smeared.csv", delimiter=',', skip_header=True).T
enu_ES_nue, ereco_ES_nue = np.genfromtxt("inputs/Enu_Ereco_ES_nue_smeared.csv", delimiter=',', skip_header=True).T
enu_CC, ereco_CC = np.genfromtxt("inputs/Enu_Ereco_CC_smeared.csv", delimiter=',', skip_header=True).T

# Load Background
ereco = np.genfromtxt("inputs/Ereco_NGamma_1e6-1_evts.csv", delimiter=',', skip_header=True).T * 1e-6

# Generate response matrices and background hist
bin_edges = np.linspace(0.1, 20, 51)*1e-3

background_CC, bin_edges = np.histogram(ereco, bins=bin_edges)
background_CC = background_CC * 2e8 / np.sum(background_CC)

response_ES_nuother, xedges, yedges = np.histogram2d(enu_ES_nuother, ereco_ES_nuother, bins=(bin_edges, bin_edges), density=False)
response_ES_nue, xedges, yedges = np.histogram2d(enu_ES_nue, ereco_ES_nue, bins=(bin_edges, bin_edges), density=False)
response_CC, xedges, yedges = np.histogram2d(enu_CC, ereco_CC, bins=(bin_edges, bin_edges), density=False)

# Normalize each column
response_ES_nuother_norm = response_ES_nuother / np.where(response_ES_nuother.sum(axis=1, keepdims=True) == 0, 1, response_ES_nuother.sum(axis=1, keepdims=True))
response_ES_nue_norm = response_ES_nue / np.where(response_ES_nue.sum(axis=1, keepdims=True) == 0, 1, response_ES_nue.sum(axis=1, keepdims=True))
response_CC_norm = response_CC / np.where(response_CC.sum(axis=1, keepdims=True) == 0, 1, response_CC.sum(axis=1, keepdims=True))

# Propagate samples
sample_ES_nue = np.matmul(response_ES_nue_norm.T, (ES_nue_8B + ES_nue_hep))
sample_ES_nuother = np.matmul(response_ES_nuother_norm.T, (ES_nuother_8B + ES_nuother_hep))
sample_CC = np.matmul(response_CC_norm.T, (CC_8B + CC_hep))


# Propagate probabilities:
solar_probs = np.array([
    [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': dm2/5 * 1e4-0.09, 'dm2_21': dm2}, energy, solar_model)
     for energy in calc_energies]
    for dm2 in dm2_21_values
])


probs_8B_ES_nue = np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, E, solar_model) for E in enu_ES_nue])
# probs_hep_ES_nue = solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, enu_ES_nue, solar_model, process='hep')

probs_8B_ES_nuother = 1 - np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, E, solar_model) for E in enu_ES_nuother])
# probs_hep_ES_nuother =  1- solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, enu_ES_nuother, solar_model, process='hep')

probs_8B_CC = np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, E, solar_model) for E in enu_CC])
# probs_hep_CC = solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': 0.022, 'dm2_21': 7.53e-5}, enu_CC, solar_model, process='hep')


dm2 = 1.5e-4
th13 = dm2/5 * 1e4-0.09

probs_ES_nue_upper = np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': th13, 'dm2_21': dm2}, E, solar_model) for E in enu_ES_nue])
probs_ES_nuother_upper = 1 - np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': th13, 'dm2_21': dm2}, E, solar_model) for E in enu_ES_nuother])
probs_CC_upper = np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': th13, 'dm2_21': dm2}, E, solar_model) for E in enu_CC])

# dm2 = 4.95e-5
# th13 = dm2/5 * 1e4-0.09

# probs_ES_nue_lower = np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': th13, 'dm2_21': dm2}, E, solar_model) for E in enu_ES_nue])
# probs_ES_nuother_lower = 1 - np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': th13, 'dm2_21': dm2}, E, solar_model) for E in enu_ES_nuother])
# probs_CC_lower = np.array([solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': th13, 'dm2_21': dm2}, E, solar_model) for E in enu_CC])


# WE DISREGARD THE HEP PROBABILITY BECAUSE THE WEIGHTING IS MINIMAL

probs_ES_nue = probs_8B_ES_nue
probs_ES_nuother = probs_8B_ES_nuother
probs_CC = probs_8B_CC

#############################################################################################################
fig, (axProbs, ax) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2]})


# Calculate solar surface probabilities for each dm2_21 and each energy
# solar_probs = np.array([
#     [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': dm2/5 * 1e4-0.1, 'dm2_21': dm2}, energy, solar_model)
#      for energy in calc_energies]
#     for dm2 in dm2_21_values
# ])


# for probs in solar_probs:
#     axProbs.plot(calc_energies*1e3, probs, linestyle='-', color='b', linewidth=1)

# solar_probs = np.array([
#     [solar_surface_probs({'sin2_th12': 0.307, 'sin2_th13': dm2/5 * 1e4-0.1, 'dm2_21': dm2}, energy, solar_model, process='hep')
#      for energy in calc_energies]
#     for dm2 in dm2_21_values
# ])

# for probs in solar_probs:
 #    axProbs.plot(calc_energies*1e3, probs, linestyle='-', color='r', linewidth=1, alpha=0.4)


plot_95_percent_contour(axProbs, ereco_ES_nuother, probs_ES_nuother_upper, 'chocolate', alpha=0.1, linestyles=[':'])
# plot_95_percent_contour(axProbs, ereco_ES_nuother, probs_ES_nuother_lower, 'chocolate', alpha=0.1, linestyles=[':'])

plot_95_percent_contour(axProbs, ereco_ES_nue, probs_ES_nue_upper, 'red', alpha=0.1, linestyles=[':'])
# plot_95_percent_contour(axProbs, ereco_ES_nue, probs_ES_nue_lower, 'red', alpha=0.1, linestyles=[':'])

plot_95_percent_contour(axProbs, ereco_CC, probs_CC_upper, 'b', alpha=0.1, linestyles=[':'])
# plot_95_percent_contour(axProbs, ereco_CC, probs_CC_lower, 'b', alpha=0.1, linestyles=[':'])

plot_95_percent_contour(axProbs, ereco_ES_nuother, probs_ES_nuother, 'chocolate', alpha=1, fill=False)
plot_95_percent_contour(axProbs, ereco_ES_nue, probs_ES_nue, 'red', alpha=1, fill=False)
plot_95_percent_contour(axProbs, ereco_CC, probs_CC, 'b', alpha=1, fill=False)


axProbs.set_ylim(0, .75)

# Plot the first three datasets on the primary y-axis
ax.step(energies, sample_ES_nuother, where='mid', label='ES_nuother', linestyle='-', color='chocolate', linewidth=2)
ax.step(energies, sample_ES_nue, where='mid', label='ES_nue', linestyle='-', color='red', linewidth=2)
ax.step(energies, sample_CC, where='mid', label='CC', linestyle='-', color='b', linewidth=2)


# Set y-axis limits for ax to start from zero
ax.set_ylim(0, ax.get_ylim()[1])

# Set x-axis limits
ax.set_xlim([energies.min(), energies.max()])

ax2 = ax.twinx()
ax2.step(energies, background_CC, where='mid', label='CC_background', linestyle='--', color='tab:green', linewidth=2)
ax2.set_ylim(0, ax2.get_ylim()[1])

ax.axvline(x=3, color='gray', linewidth=1)
axProbs.axvline(x=3, color='gray', linewidth=1)

axProbs.axvline(x=5, ymin=0.35, ymax=0.6, color='tab:green', linewidth=2.5, linestyle='--')

ymin, ymax = ax.get_ylim()
ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=3, color='gray', alpha=0.15)

ymin, ymax = axProbs.get_ylim()
axProbs.fill_betweenx(y=[ymin, ymax], x1=0, x2=3, color='gray', alpha=0.15)


# Set y-axis limits for ax2 to start from zero and be twice as large
#ax2.set_ylim(0, ax2.get_ylim()[1] * 2)

# Add labels and legend
ax.set_xlabel(r'$E_{rec}$ (MeV)')
ax.set_ylabel(r'Reconstructed events')
axProbs.set_ylabel(r'$P_{eff}$')

# Position legends outside the plot
ax.legend(loc='upper right')
ax2.legend(loc='lower right')
plt.tight_layout()

# Show the plot
plt.savefig('images/oscillated_sample_ereco.png', dpi=300)
