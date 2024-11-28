include("../src/objects.jl")

function mswProb(oscpars, E_true, n_e)
    @inbounds begin
        # Calculate the cosine of twice the angle theta_12 using the oscillation parameter
        c2th12 = cos.(2 .* asin.(sqrt.(oscpars.sin2_th12)))  # Use element-wise operations
    
        # Calculate the beta parameter, which is a function of electron density, Fermi constant, energy, and oscillation parameters
        Acc = 2 .* sqrt(2) .* n_e .* G_f .* E_true .* (1 .- oscpars.sin2_th13)  # Ensure element-wise multiplication
        beta = Acc ./ oscpars.dm2_21  # Ensure element-wise division
    
        # Calculate the modified cosine of twice the angle theta_12 in matter
        c2th12m = (c2th12 .- beta) ./ sqrt.((c2th12 .- beta).^2 .+ (1 .- c2th12.^2))  # Ensure element-wise operations
    
        # Calculate the modified sine squared of theta_13 in matter
        s13m = sqrt.(oscpars.sin2_th13)  # Use element-wise operation if sin2_th13 is a vector
    
        # Calculate the probability for the electron neutrino flavor
        Probs = 1 ./ 2 .* (1 .- oscpars.sin2_th13) .* (1 .- s13m.^2) .* (1 .+ c2th12 .* c2th12m) .+ oscpars.sin2_th13 .* s13m.^2  # Ensure element-wise operations
    end

    # Convert the mutable vector of probabilities to an immutable static vector and return it
    return Probs
end



function peanutsProb(params, E_true::Float64, n_e_array::Vector{Float64})::Vector{Float64}

    # Precompute quantities that do not depend on n_e
    th13 = asin(sqrt(params.sin2_th13))
    c2th13 = cos(2 * th13)
    c2th12 = cos(2 * asin(sqrt(params.sin2_th12)))
    m_ee = (1 - params.sin2_th12) * (params.dm2_21 + m32) + params.sin2_th12 * m32

    # Check for division by zero
    if m_ee == 0
        throw(Error("m_ee is zero, leading to division by zero."))
    end

    # Calculate Acc for each n_e using broadcasting
    @inbounds Acc = 2 * sqrt(2) * E_true * G_f .* n_e_array

    # Calculate the modified cosine of twice the angle theta_13 in matter
    c2th13m = (c2th13 .- Acc ./ m_ee) ./ sqrt.((c2th13 .- Acc ./ m_ee) .^ 2 .+ 1 .- c2th13^2)

    # Calculate the modified mixing angle theta_13 in matter
    th13m = 1 / 2 .* acos.(c2th13m)

    # Calculate the modified effective potential
    Accm = Acc .* cos.(th13m) .^ 2 .+ m_ee .* sin.(th13m .- th13) .^ 2

    # Calculate the modified cosine of twice the angle theta_12 in matter
    c2th12m = (c2th12 .- Accm ./ params.dm2_21) ./ sqrt.((c2th12 .- Accm ./ params.dm2_21) .^ 2 .+ (1 .- c2th12^2) .* cos.(th13m .- th13) .^ 2)

    # Calculate the probability for the electron neutrino flavor
    electron_probs = 1 / 2 .* (1 .- params.sin2_th13) .* (1 .- sin.(th13m) .^ 2) .* (1 .+ c2th12 .* c2th12m) .+ params.sin2_th13 .* sin.(th13m) .^ 2

    return electron_probs
end


function solarSurfaceProbs(E_true::Float64, params, solarModel; process="8B")
    # Calculate the neutrino oscillation probabilities using the electron density
    enuOscProb = mswProb(params, E_true, solarModel.n_e)

    # Select the appropriate production fraction based on the process
    prodFraction = process == "8B" ? solarModel.prodFractionBoron :
                   process == "hep" ? solarModel.prodFractionHep :
                   error("Invalid process specified. Please use '8B' or 'hep'.")

    # Calculate the weighted sum and total weight
    weighted_sum = sum(prodFraction .* enuOscProb)
    total_weight = sum(prodFraction)

    # Compute the probability
    prob_nue = weighted_sum / total_weight

    return prob_nue
end

function solarSurfaceProbs_approx(E_true::Float64, params, solarModel; process="8B")
    # Select the appropriate average density based on the process
    n_e = process == "8B" ? solarModel.avgNeBoron :
          process == "hep" ? solarModel.avgNeHep :
          error("Invalid process specified. Please use '8B' or 'hep'.")

    # Calculate the neutrino oscillation probabilities using the electron density
    enuOscProb = mswProb(params, E_true, n_e)

    return enuOscProb
end

function averageProbOverBins(bin_edges::Vector{Float64}, params, solarModel; process="8B")
    # Pre-allocate the bin_probs array
    bin_probs = Vector{Float64}(undef, length(bin_edges) - 1)

    # Iterate over each pair of bin edges
    for i in 1:(length(bin_edges) - 1)
        emin = bin_edges[i]
        emax = bin_edges[i + 1]

        # Define the integrand function for the current bin
        integrand(E_true) = solarSurfaceProbs(E_true, params, solarModel; process=process)

        # Perform the integration over the current bin
        integral, _ = quadgk(integrand, emin, emax, atol=1e-8, rtol=1e-8)
        average = integral / (emax - emin)

        # Store the result in the bin_probs array
        @inbounds bin_probs[i] = average
    end

    return bin_probs
end

function centralProbOverBins(bin_edges::Vector{Float64}, params, solarModel; process="8B")
    # Pre-allocate the bin_probs array
    bin_probs = Vector{Float64}(undef, length(bin_edges) - 1)
    bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2.0

    # Evaluate solarSurfaceProbs at all bin centers using broadcasting
    bin_probs = solarSurfaceProbs_approx.(bin_centers, Ref(params), Ref(solarModel); process=process)

    return bin_probs
end



# TESTING
using Test

function test_solar_surface_probs_8b_valid_inputs()
    # Assuming OscillationParameters and a mock solarModel are defined elsewhere
    params = OscillationParameters()  # Fill with appropriate values
    E_true = 10.0  # Example energy value

    # Call the function with "8B" process
    @time prob_nue = solarSurfaceProbs(params, E_true, solarModel, process="8B")


end

