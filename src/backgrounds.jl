
include("../src/histHelpers.jl")

# Extract background files as dataframes
df_ES_list = extract_dataframes(ES_filepaths_BG)
df_CC_list = extract_dataframes(CC_filepaths_BG)

# Get bin heights and central bin energies for the backgrounds
ES_bg = []


for df in df_ES_list
    if "weights" in names(df)
        ES_temp, ES_temp_etrue = create_histogram(df.Ereco, Ereco_bins_ES_extended, weights = df.weights)
    else
        ES_temp, ES_temp_etrue = create_histogram(df.Ereco, Ereco_bins_ES_extended)
    end
    push!(ES_bg, ES_temp)
end


CC_bg = []
for df in df_CC_list
    if "weights" in names(df)
        CC_temp, CC_temp_etrue = create_histogram(df.Ereco, Ereco_bins_CC_extended, weights = df.weights)
    else
        CC_temp, CC_temp_etrue = create_histogram(df.Ereco, Ereco_bins_CC_extended)
    end
    push!(CC_bg, CC_temp)
end

# Normalise accordingly and add systematic as nuisance parameters if needed
global ES_bg_norms_true = []
global ES_bg_norms_pars = []
global ES_bg_par_counts = []

global CC_bg_norms_true = []
global CC_bg_norms_pars = []
global CC_bg_par_counts = []

for (bg, norm, sys) in zip(ES_bg, ES_bg_norms, ES_bg_sys)
    if sys == 0
        bg .*= norm
        push!(ES_bg_par_counts, 0)
    else
        push!(ES_bg_norms_true, norm)
        push!(ES_bg_norms_pars, Truncated(Normal(norm, norm * sys), 0.0, norm*2))
        push!(ES_bg_par_counts, 1)
    end
end

for (bg, norm, sys) in zip(CC_bg, CC_bg_norms, CC_bg_sys)
    if sys == 0
        bg .*= norm
        push!(CC_bg_par_counts, 0)
    else
        push!(CC_bg_norms_true, norm)
        push!(CC_bg_norms_pars, Truncated(Normal(norm, norm * sys), 0.0, norm*2))
        push!(CC_bg_par_counts, 1)
    end
end