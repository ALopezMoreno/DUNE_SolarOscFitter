#=
prem.jl — PREM (Dziewonski & Anderson 1981) electron density with the ocean replaced by upper crust,
as the nominal Earth of parchita/EarthModels (theta = 0). Radii in km, density in g/cm³.
Selected with `earth_model_file: "PREM"`.
=#

if !@isdefined(PREM_RADIUS_KM)

const PREM_RADIUS_KM = 6371.0

# Radii where the density jumps, ascending: ICB, CMB, 670, 400, 220, Moho, upper/lower crust
const PREM_BOUNDARIES_KM = [1221.5, 3480.0, 5701.0, 5971.0, 6151.0, 6346.6, 6356.0]
const PREM_R_TZ = 5771.0   # change of polynomial inside the transition zone (no jump)

# Electron fraction per region between the boundaries, innermost first
const PREM_YE = [0.466, 0.466, 0.496, 0.496, 0.496, 0.496, 0.497, 0.497]

"""
    prem_region(r) -> Int

Region of radius `r` [km] between `PREM_BOUNDARIES_KM`: 1 = inner core, 8 = upper crust.
A radius on a boundary belongs to the region above it.
"""
prem_region(r::Real) = searchsortedlast(PREM_BOUNDARIES_KM, r) + 1

"""
    prem_density(r, region=prem_region(r))

PREM density [g/cm³] at radius `r` [km], continental crust instead of ocean. `region` picks the
side of a boundary.
"""
function prem_density(r::Real, region::Int=prem_region(r))
    x = r / PREM_RADIUS_KM
    region == 1 && return 13.0885 - 8.8381x^2
    region == 2 && return 12.5815 - 1.2638x - 3.6426x^2 - 5.5281x^3
    region == 3 && return 7.9565 - 6.4761x + 5.5283x^2 - 3.0807x^3
    region == 4 && return r < PREM_R_TZ ? 5.3197 - 1.4836x : 11.2494 - 8.0298x
    region == 5 && return 7.1089 - 3.8045x
    region == 6 && return 2.6910 + 0.6924x
    region == 7 && return 2.900
    return 2.600
end

"""
    prem_electron_density(r, region=prem_region(r))

n_e = ρ Y_e [N_A/cm³] at radius `r` [km].
"""
prem_electron_density(r::Real, region::Int=prem_region(r)) = prem_density(r, region) * PREM_YE[region]

"""
    prem_table(; step_km=1.0, jump_half_width_km=1e-9) -> (radius, density, electron_fraction)

PREM sampled every `step_km` at most within each region, radius ascending in km. Each jump is a
step of width 2 `jump_half_width_km` around its boundary, so linear interpolation keeps it sharp.
"""
function prem_table(; step_km::Real=1.0, jump_half_width_km::Real=1e-9)
    edges = vcat(0.0, PREM_BOUNDARIES_KM, PREM_RADIUS_KM)
    r, rho, ye = Float64[], Float64[], Float64[]
    for k in 1:length(edges)-1
        lo = k == 1 ? edges[k] : edges[k] + jump_half_width_km
        hi = k == length(edges) - 1 ? edges[k+1] : edges[k+1] - jump_half_width_km
        for ri in range(lo, hi; length=max(2, ceil(Int, (hi - lo) / step_km) + 1))
            push!(r, ri); push!(rho, prem_density(ri, k)); push!(ye, PREM_YE[k])
        end
    end
    return r, rho, ye
end

end # @isdefined
