
#=
backgrounds.jl

Background event processing for the Solar Oscillation Fitter.
This module loads and processes background Monte Carlo samples for both
Elastic Scattering (ES) and Charged Current (CC) detection channels.

Key Features:
- Background MC sample loading and histogram creation
- Detection efficiency calculations
- Systematic uncertainty handling for background normalizations
- Support for weighted and unweighted MC samples
- Automatic parameter setup for MCMC fitting

The backgrounds are normalized to the expected detection time and exposure,
with optional systematic uncertainties treated as nuisance parameters.

Author: [Author name]
=#

include("../src/histHelpers.jl")

# Load background Monte Carlo samples as dataframes
df_ES_list = extract_dataframes(ES_filepaths_BG)  # ES background samples
df_CC_list = extract_dataframes(CC_filepaths_BG)  # CC background samples


# Process ES background samples
ES_bg = []
ES_sides = []
for df in df_ES_list
    # Create histograms with or without MC weights
    if "weights" in names(df)
        # Weighted MC samples
        ES_temp, ES_temp_etrue = create_histogram(df.Ereco[df.mask], Ereco_bins_ES_extended, weights=df.weights[df.mask], normalise=true)
        ES_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_ES_extended, weights=df.weights[df.mask], normalise=false)
        ES_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_ES_extended, weights=df.weights, normalise=false)
    else
        # Unweighted MC samples
        ES_temp, ES_temp_etrue = create_histogram(df.Ereco, Ereco_bins_ES_extended, normalise=true)
        ES_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_ES_extended, normalise=false)
        ES_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_ES_extended, normalise=false)
    end

    if "side" in names(df)
        side = df.side
    else
        side = -1
    end

    # Calculate detection efficiency: selected events / total events
    ES_eff_bg =  @. ifelse(ES_temp_total == 0, 0.0, ES_temp_selec / ES_temp_total)
    
    # Scale by detection time, efficiency, and exposure normalization and attenuation ratio over the 50M events of the MC
    attenuation = sum(ES_temp_total) / 50e6

    push!(ES_bg, ES_temp .* detection_time .* ES_eff_bg .* ES_normalisation .* attenuation)
    push!(ES_sides, side)
end


# Process CC background samples
CC_bg = []
for df in df_CC_list
    # Create histograms with or without MC weights
    if "weights" in names(df)
        # Weighted MC samples
        CC_temp, CC_temp_etrue = create_histogram(df.Ereco, Ereco_bins_CC_extended, weights=df.weights, normalise=true)
        CC_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_CC_extended, weights=df.weights[df.mask], normalise=false)
        CC_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_CC_extended, weights=df.weights, normalise=false)
    else
        # Unweighted MC samples
        CC_temp, CC_temp_etrue = create_histogram(df.Ereco, Ereco_bins_CC_extended, normalise=true)
        CC_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_CC_extended, normalise=false)
        CC_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_CC_extended, normalise=false)        
    end

    # Calculate detection efficiency: selected events / total events
    CC_eff_bg = @. ifelse(CC_temp_total == 0, 0.0, CC_temp_selec / CC_temp_total)
    
    # Scale by detection time, efficiency, and exposure normalization
    # push!(CC_bg, CC_temp .* detection_time .* CC_eff_bg .* CC_normalisation)
    ## MC IS ALREADY NORMALISED TO 1 KT YEAR!
    push!(CC_bg, CC_temp_selec .* 10 .* CC_normalisation)
end

# Setup systematic uncertainties for background normalizations
# These will be treated as nuisance parameters in the MCMC fit

global ES_bg_norms_true = []  # True normalization values for ES backgrounds
global ES_bg_norms_pars = []  # Prior distributions for ES background systematics
global ES_bg_par_counts = []  # Count of systematic parameters per ES background

global CC_bg_norms_true = []  # True normalization values for CC backgrounds
global CC_bg_norms_pars = []  # Prior distributions for CC background systematics
global CC_bg_par_counts = []  # Count of systematic parameters per CC background

# Apply normalizations and setup systematic parameters for ES backgrounds
if ES_mode
    for (bg, norm, sys) in zip(ES_bg, ES_bg_norms, ES_bg_sys)
        if sys == 0
            # No systematic uncertainty - apply fixed normalization
            bg .*= norm
            push!(ES_bg_par_counts, 0)
        else
            # Systematic uncertainty present - create nuisance parameter
            push!(ES_bg_norms_true, norm)
            # Truncated normal prior: mean=norm, std=norm*sys, bounds=[0, 2*norm]
            push!(ES_bg_norms_pars, Truncated(Normal(norm, norm * sys), 0.0, norm*2))
            push!(ES_bg_par_counts, 1)
        end
    end
end

# Apply normalizations and setup systematic parameters for CC backgrounds
if CC_mode
    for (bg, norm, sys) in zip(CC_bg, CC_bg_norms, CC_bg_sys)
        if sys == 0
            # No systematic uncertainty - apply fixed normalization
            # The MC for CC was created using a 4pi flux of 9.86e-8 n/cm2/s, which corresponds to a flux on the wall of 2.2e-6 n/cm2/s
            bg .*= norm / 2.2e-6
            push!(CC_bg_par_counts, 0)
        else
            # Systematic uncertainty present - create nuisance parameter
            push!(CC_bg_norms_true, norm)
            # Truncated normal prior: mean=norm, std=norm*sys, bounds=[0, 2*norm]
            push!(CC_bg_norms_pars, Truncated(Normal(norm, norm * sys), 0.0, norm*2))
            push!(CC_bg_par_counts, 1)
        end
    end
end