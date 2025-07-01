
include("../src/histHelpers.jl")

# Extract background files as dataframes
df_ES_list = extract_dataframes(ES_filepaths_BG)
df_CC_list = extract_dataframes(CC_filepaths_BG)


ES_bg = []
for df in df_ES_list
    if "weights" in names(df)
        ES_temp, ES_temp_etrue = create_histogram(df.Ereco, Ereco_bins_ES_extended, weights=df.weights, normalise=true)
        ES_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_ES_extended, weights=df.weights[df.mask], normalise=false)
        ES_temp_total, _ = create_histogram(df.Eraw, Ereco_bins_ES_extended, weights=df.weights, normalise=false)
    else
        ES_temp, ES_temp_etrue = create_histogram(df.Ereco, Ereco_bins_ES_extended, normalise=true)
        ES_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_ES_extended, normalise=false)
        ES_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_ES_extended, normalise=false)
    end

    ES_eff_bg =  @. ifelse(ES_temp_total == 0, 0.0, ES_temp_selec / ES_temp_total)
    push!(ES_bg, ES_temp .* detection_time .* ES_eff_bg .* ES_normalisation)
end


CC_bg = []
for df in df_CC_list
    if "weights" in names(df)
        CC_temp, CC_temp_etrue = create_histogram(df.Ereco, Ereco_bins_CC_extended, weights=df.weights, normalise=true)
        CC_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_CC_extended, weights=df.weights[df.mask], normalise=false)
        CC_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_CC_extended, weights=df.weights, normalise=false)
    else
        CC_temp, CC_temp_etrue = create_histogram(df.Ereco, Ereco_bins_CC_extended, normalise=true)
        CC_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_CC_extended, normalise=false)
        CC_temp_total, _ = create_histogram(df.Ereco, Ereco_bins_CC_extended, normalise=false)        
    end

    CC_eff_bg = @. ifelse(CC_temp_total == 0, 0.0, CC_temp_selec / CC_temp_total)
    push!(CC_bg, CC_temp .* detection_time .* CC_eff_bg .* CC_normalisation)
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