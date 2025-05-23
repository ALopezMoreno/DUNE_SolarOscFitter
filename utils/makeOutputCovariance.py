import numpy as np
import time
import sys
import argparse
import matplotlib.pyplot as plt
import h5py
import posteriorHelpers
import plotting
import cmasher

parser = argparse.ArgumentParser(description="Create a covariance matrix from one or several MCMC chains.")

parser.add_argument('chains', nargs='+', help="Input MCMC chain files.")
parser.add_argument('-o', '--output', type=str, help="Output file (optional).")
parser.add_argument('-p', '--plot', action="store_true", help="plot covariance matrix?")

args = parser.parse_args()

mcmc_chains = args.chains 
output_name = args.output
plot_cov = args.plot

# Check if at least one file is provided
if not mcmc_chains:
    print("Error: No MCMC output files provided.")
    sys.exit(1)


#############################################################
################ ----- Begin execution ----- ################
#############################################################

start_time = time.time()

data = posteriorHelpers.load_posterior(mcmc_chains, 'all')

# remove metadata
weights = data.pop('weights', None)
chains = data.pop('chains', None)
steps = data.pop('stepno', None)

a = data.pop('a', None)
l = data.pop('l', None)

new_data =  np.array(list(data.values()))

variables = list(data.keys())

covariance = np.cov(new_data, fweights=weights)

# Save the covariance matrix to a JLD2 file
if output_name is None:
        print("saving output as images/posterior_covariance.jld2")
        with h5py.File('outputs/posterior_covariance.jld2', 'w') as f:
            f.create_dataset('posterior_cov', data=covariance)
else:
    print("saving output as " + output_name + "_cov.jld2")
    with h5py.File(output_name + "_cov.jld2", 'w') as f:
        f.create_dataset('posterior_cov', data=covariance)

if plot_cov:
    fig, ax = plotting.plot_covariance(covariance, labels=variables, colormap='coolwarm')
    if output_name is None:
        print("saving image as images/posterior_covariance.pdf")
        fig.savefig("images/posterior_covariance.pdf", dpi=300, format='pdf')
    else:
        print("saving image as " + output_name + ".pdf")
        fig.savefig(output_name + ".pdf", dpi=300, format='pdf')