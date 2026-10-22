#=
tabulated.jl — one validated helper for tabulated model inputs (SPEC 8.1).

    t = TabulatedInput(x, y; name, source, units, extrapolation = :error, nonnegative = false)
    t(q)            # linear interpolation (Interpolations.jl) at a scalar or each element of a vector
    domain(t)       # (x[1], x[end])
    used_range(t)   # (min, max) queried so far

Construction validates the table (validate_table). Outside the domain, `:error` throws a
TabulatedDomainError, `:flat` returns the end value, `:zero` returns 0; queries outside are counted
in every mode and named inputs are listed by log_tabulated_inputs() after setup. Setup only: not
type-stable for Enzyme and it mutates its usage record. `read_numeric_table` reads delimited numeric
tables with a required or forbidden header row.
=#
using Interpolations
using Logging
using Printf

struct TabulatedDomainError <: Exception
    name::String
    source::String
    x::Float64
    domain::Tuple{Float64,Float64}
    units::String
end

function Base.showerror(io::IO, e::TabulatedDomainError)
    u = isempty(e.units) ? "" : " $(e.units)"
    src = isempty(e.source) ? "" : " ($(e.source))"
    print(io, "TabulatedDomainError: $(e.name)$(src) evaluated at x = $(e.x)$(u), outside its table ",
          "[$(e.domain[1]), $(e.domain[2])]$(u); it is never extrapolated (extrapolation = :error)")
end

struct TabulatedInput{I}
    name::String
    source::String
    units::String                 # unit of x, for messages and the setup log
    x::Vector{Float64}
    y::Vector{Float64}
    extrapolation::Symbol         # :error, :flat or :zero
    itp::I
    used::Base.RefValue{NTuple{2,Float64}}   # (min, max) queried
    n_outside::Base.RefValue{NTuple{2,Int}}  # queries below / above the domain
end

const TABULATED_EXTRAPOLATIONS = (:error, :flat, :zero)

_where(name, source) = isempty(source) ? name : "$name ($source)"

"""
    validate_table(x, y; name, source = "", nonnegative = false)

Error unless x and y have equal lengths ≥ 2, all values are finite, x is strictly increasing
and (with `nonnegative`) y ≥ 0. Messages name the input and the first offending row.
"""
function validate_table(x::AbstractVector, y::AbstractVector; name::AbstractString, source::AbstractString = "",
                        nonnegative::Bool = false)
    w = _where(name, source)
    length(x) == length(y) || error("$w: $(length(x)) nodes but $(length(y)) values")
    length(x) >= 2 || error("$w: a table needs at least 2 points, it has $(length(x))")
    i = findfirst(!isfinite, x)
    i === nothing || error("$w: non-finite node x[$i] = $(x[i])")
    i = findfirst(!isfinite, y)
    i === nothing || error("$w: non-finite value y[$i] = $(y[i]) at x = $(x[i])")
    i = findfirst(k -> !(x[k+1] > x[k]), 1:length(x)-1)
    i === nothing ||
        error("$w: nodes must be strictly increasing (sorted, no duplicates); x[$i] = $(x[i]), x[$(i+1)] = $(x[i+1])")
    if nonnegative
        i = findfirst(<(0), y)
        i === nothing || error("$w: negative value y[$i] = $(y[i]) at x = $(x[i])")
    end
    return nothing
end

if !@isdefined(TABULATED_REGISTRY)
    # Named inputs, by name, for the setup log (a later input with the same name replaces it)
    const TABULATED_REGISTRY = Dict{String,Any}()
end

function TabulatedInput(x::AbstractVector, y::AbstractVector; name::AbstractString, source::AbstractString = "",
                        units::AbstractString = "", extrapolation::Symbol = :error, nonnegative::Bool = false,
                        register::Bool = true)
    extrapolation in TABULATED_EXTRAPOLATIONS ||
        error("$(_where(name, source)): extrapolation must be one of $(TABULATED_EXTRAPOLATIONS), got :$extrapolation")
    validate_table(x, y; name, source, nonnegative)
    xs, ys = Vector{Float64}(x), Vector{Float64}(y)
    itp = linear_interpolation(xs, ys; extrapolation_bc = Throw())
    t = TabulatedInput(String(name), String(source), String(units), xs, ys, extrapolation, itp,
                       Ref((Inf, -Inf)), Ref((0, 0)))
    register && (TABULATED_REGISTRY[t.name] = t)
    return t
end

domain(t::TabulatedInput) = (t.x[1], t.x[end])
used_range(t::TabulatedInput) = t.used[][1] <= t.used[][2] ? t.used[] : (NaN, NaN)
n_outside(t::TabulatedInput) = t.n_outside[]

"""
    reset_usage!(t)

Forget the recorded query range and out-of-domain counts.
"""
function reset_usage!(t::TabulatedInput)
    t.used[] = (Inf, -Inf); t.n_outside[] = (0, 0)
    return t
end

function (t::TabulatedInput)(q::Real)
    qf = Float64(q)
    lo, hi = t.x[1], t.x[end]
    u = t.used[]
    t.used[] = (min(u[1], qf), max(u[2], qf))
    lo <= qf <= hi && return t.itp(q)
    below = qf < lo
    n = t.n_outside[]
    t.n_outside[] = below ? (n[1] + 1, n[2]) : (n[1], n[2] + 1)
    t.extrapolation === :error && throw(TabulatedDomainError(t.name, t.source, qf, (lo, hi), t.units))
    t.extrapolation === :flat && return t.itp(below ? lo : hi)
    return zero(t.itp(lo))
end

(t::TabulatedInput)(q::AbstractVector) = [t(v) for v in q]

"""
    check_covers(t, a, b; what, rtol = 1e-9)

Error unless [a, b] lies inside the domain of `t`. `rtol` (relative to the domain width)
absorbs the rounding of computed bin edges; `what` names the range in the message.
"""
function check_covers(t::TabulatedInput, a::Real, b::Real; what::AbstractString, rtol::Real = 1e-9)
    lo, hi = domain(t)
    tol = rtol * (hi - lo)
    (lo - tol <= a && b <= hi + tol) ||
        error("$what [$a, $b] is outside $(_where(t.name, t.source)), which covers [$lo, $hi] $(t.units)")
    return nothing
end

"""
    tabulated_summary(inputs = registered) -> Vector{String}

One line per input: name, source, extrapolation policy, domain, range queried so far, and the
number of queries outside the domain.
"""
function tabulated_summary(inputs = sort!(collect(values(TABULATED_REGISTRY)); by = t -> t.name))
    lines = String[]
    for t in inputs
        lo, hi = domain(t)
        ulo, uhi = used_range(t)
        nb, na = n_outside(t)
        used = isnan(ulo) ? "not queried" : @sprintf("[%.10g, %.10g]", ulo, uhi)
        out = nb + na == 0 ? "none outside" : "$nb below, $na above the table"
        push!(lines, @sprintf("  %-28s  %-6s  domain [%.10g, %.10g] %s, used %s, %s  (%s)",
                              t.name, ":" * String(t.extrapolation), lo, hi, t.units, used, out, t.source))
    end
    return lines
end

# Off the likelihood path: called once at the end of setup.
@noinline function log_tabulated_inputs(inputs = sort!(collect(values(TABULATED_REGISTRY)); by = t -> t.name))
    @logmsg LogLevel(2) join(vcat("Tabulated inputs (domain vs range used at setup):", tabulated_summary(inputs)), "\n")
    return nothing
end

"""
    read_numeric_table(path; name, ncols, header::Bool, delim = ',') -> Vector{Vector{Float64}}

Columns of a numeric table. '#' comment and blank lines are skipped. `header = true` requires a
header row (a numeric first row is an error); `header = false` forbids one (a non-numeric first
row is an error). `delim = nothing` splits on whitespace. Every data row must have `ncols` numbers.
"""
function read_numeric_table(path::AbstractString; name::AbstractString, ncols::Int, header::Bool, delim = ',')
    isfile(path) || error("$name: table not found: $path")
    rows = [strip(l) for l in eachline(path)]
    rows = [l for l in rows if !isempty(l) && !startswith(l, "#")]
    isempty(rows) && error("$name: table $path is empty")
    split_row(l) = delim === nothing ? split(l) : strip.(split(l, delim))
    numeric(l) = all(f -> tryparse(Float64, f) !== nothing, split_row(l))
    if header
        numeric(rows[1]) &&
            error("$name: table $path has no header row (first row is numeric: \"$(rows[1])\"); add a header row")
        length(split_row(rows[1])) == ncols ||
            error("$name: table $path: expected $ncols columns, header has $(length(split_row(rows[1])))")
        rows = rows[2:end]
    else
        isempty(rows) || numeric(rows[1]) ||
            error("$name: table $path has a non-numeric first row (\"$(rows[1])\"); it is read without a header")
    end
    cols = [Float64[] for _ in 1:ncols]
    for (i, l) in enumerate(rows)
        f = split_row(l)
        vals = length(f) == ncols ? tryparse.(Float64, f) : nothing
        (vals === nothing || any(isnothing, vals)) &&
            error("$name: table $path: data row $i is not $ncols numbers: \"$l\"")
        for c in 1:ncols
            push!(cols[c], vals[c])
        end
    end
    return cols
end
