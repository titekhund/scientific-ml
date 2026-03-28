# exp_realdata_viz.jl — Download macro data and produce two publication-quality figures.
# Self-contained: no dependency on the ScientificML module.

using Pkg
for p in ["XLSX", "CairoMakie", "Downloads"]
    p in keys(Pkg.project().dependencies) || haskey(Pkg.dependencies(), findfirst(d -> d.name == p, Pkg.dependencies())) || try
        @eval using $(Symbol(p))
    catch
        Pkg.add(p)
    end
end

using Downloads, XLSX, CairoMakie, Dates

# ── Download & read data ─────────────────────────────────────────────
const URL = "https://raw.githubusercontent.com/titekhund/scientific_ml_tato/main/data.xlsx"
const DATAFILE = joinpath(@__DIR__, "results", "realdata_viz", "data.xlsx")
const OUTDIR   = joinpath(@__DIR__, "results", "realdata_viz")

isfile(DATAFILE) || Downloads.download(URL, DATAFILE)

xf    = XLSX.readxlsx(DATAFILE)
sheet = xf[XLSX.sheetnames(xf)[1]]
dt    = XLSX.gettable(sheet)
colnames = dt.column_labels  # Vector{Symbol}
coldata  = dt.data           # Vector{Vector}

# Build name → vector mapping
cols = Dict(colnames[i] => coldata[i] for i in eachindex(colnames))

# Identify columns (robust to slight naming variations)
function find_col(cols, patterns...)
    ks = collect(keys(cols))
    for pat in patterns
        idx = findfirst(k -> occursin(pat, lowercase(string(k))), ks)
        idx !== nothing && return ks[idx]
    end
    error("Column not found for patterns: $patterns. Available: $ks")
end

date_key = find_col(cols, "date", "period", "quarter")
ws_key   = find_col(cols, "wage_share", "wageshare", "wage share")
ur_key   = find_col(cols, "unrate", "unemployment", "urate")
tcu_key  = find_col(cols, "tcu", "capacity", "caputil")

raw_dates = cols[date_key]
raw_ws    = cols[ws_key]
raw_ur    = cols[ur_key]
raw_tcu   = cols[tcu_key]

N = length(raw_dates)

# Parse dates (handle Date, DateTime, String, or Float64-year)
function parse_date(x)
    x isa Date && return x
    x isa DateTime && return Date(x)
    x isa AbstractString && return Date(x)
    error("Unexpected date type: $(typeof(x)): $x")
end

dates = parse_date.(raw_dates)

# Convert to Float64, keeping missings
tofloat(x::Number) = Float64(x)
tofloat(::Missing) = missing

u_all   = tofloat.(raw_ws) ./ 100          # wage share
v_all   = 1.0 .- tofloat.(raw_ur) ./ 100   # employment rate
tcu_all = tofloat.(raw_tcu)                 # capacity utilization

# ── Figure 1: Time-series ────────────────────────────────────────────
set_theme!(theme_minimal())
update_theme!(
    fontsize = 11,
    Axis = (
        xgridvisible = false, ygridvisible = false,
        topspinevisible = false, rightspinevisible = false,
        spinewidth = 0.6,
    ),
)

fig1 = Figure(size = (700, 480))

# Top panel: v (left axis) + tcu (right axis)
ax_v = Axis(fig1[1, 1]; ylabel = L"v", xlabelvisible = false)
ax_tcu = Axis(fig1[1, 1]; ylabel = L"\mathrm{TCU}", yaxisposition = :right,
              xlabelvisible = false, xticklabelsvisible = false, xticksvisible = false,
              xgridvisible = false, ygridvisible = false,
              topspinevisible = false, leftspinevisible = false, spinewidth = 0.6)
hidexdecorations!(ax_v, ticks = false, ticklabels = false)

# Date → float for plotting
date_num = Dates.value.(dates) .|> Float64

# Tick positions: every 10 years
yr_start = year(dates[1])
yr_end   = year(dates[end])
tick_years = (div(yr_start, 10) * 10 + 10):10:(div(yr_end, 10) * 10)
tick_vals  = [Float64(Dates.value(Date(y, 1, 1))) for y in tick_years]
tick_labs  = string.(tick_years)

# v series (all points, skip missing)
vm = .!ismissing.(v_all)
lines!(ax_v, date_num[vm], collect(skipmissing(v_all[vm])); color = :steelblue, linewidth = 1.0,
       label = L"v")

# tcu series (skip missing)
tm = .!ismissing.(tcu_all)
lines!(ax_tcu, date_num[tm], collect(skipmissing(tcu_all[tm])); color = :firebrick,
       linewidth = 1.0, linestyle = :dash, label = L"\mathrm{TCU}")

# Synchronise x-limits
linkxaxes!(ax_v, ax_tcu)
ax_v.xticks  = (tick_vals, tick_labs)

# Combined legend
leg_elems = [LineElement(color = :steelblue, linewidth = 1.2),
             LineElement(color = :firebrick, linewidth = 1.2, linestyle = :dash)]
Legend(fig1[1, 2], leg_elems, [L"v", L"\mathrm{TCU}"], framevisible = false, rowgap = 2)

# Bottom panel: u
ax_u = Axis(fig1[2, 1]; ylabel = L"u")
um = .!ismissing.(u_all)
lines!(ax_u, date_num[um], collect(skipmissing(u_all[um])); color = :black, linewidth = 1.0)
ax_u.xticks = (tick_vals, tick_labs)
linkxaxes!(ax_v, ax_u)

rowgap!(fig1.layout, 8)
save(joinpath(OUTDIR, "timeseries.png"), fig1; px_per_unit = 3)
println("Saved timeseries.png")

# ── Figure 2: Phase portraits by decade ──────────────────────────────
decade_ranges = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
decade_colors = Dict(
    1950 => :royalblue,   1960 => :darkorange, 1970 => :forestgreen,
    1980 => :firebrick,   1990 => :mediumpurple, 2000 => :goldenrod,
    2010 => :deeppink,    2020 => :teal,
)
decade_label(d) = "$(d)s"

# Group indices by decade; include last obs of previous decade
function decade_indices(yrs, decade_start)
    first_prev = findlast(y -> y < decade_start, yrs)
    core = findall(y -> decade_start <= y < decade_start + 10, yrs)
    isempty(core) && return Int[]
    first_prev !== nothing ? vcat(first_prev, core) : core
end

years_all = year.(dates)

fig2 = Figure(size = (900, 420))

# Left: (v, u) classical
ax_cl = Axis(fig2[1, 1]; xlabel = L"v", ylabel = L"u")
for d in decade_ranges
    idx = decade_indices(years_all, d)
    valid = [i for i in idx if !ismissing(v_all[i]) && !ismissing(u_all[i])]
    isempty(valid) && continue
    lines!(ax_cl, Float64.(collect(v_all[valid])), Float64.(collect(u_all[valid]));
           color = decade_colors[d], linewidth = 1.0, label = decade_label(d))
end

# Right: (tcu, u) structuralist — drop missing tcu
ax_st = Axis(fig2[1, 2]; xlabel = L"\mathrm{TCU}", ylabel = L"u")
for d in decade_ranges
    idx = decade_indices(years_all, d)
    valid = [i for i in idx if !ismissing(tcu_all[i]) && !ismissing(u_all[i])]
    isempty(valid) && continue
    lines!(ax_st, Float64.(collect(tcu_all[valid])), Float64.(collect(u_all[valid]));
           color = decade_colors[d], linewidth = 1.0, label = decade_label(d))
end

# Shared legend at bottom
elems = [LineElement(color = decade_colors[d], linewidth = 1.2) for d in decade_ranges]
labs  = [decade_label(d) for d in decade_ranges]
Legend(fig2[2, 1:2], elems, labs; orientation = :horizontal, framevisible = false,
       nbanks = 1, colgap = 14)
rowgap!(fig2.layout, 8)

save(joinpath(OUTDIR, "phase_decades.png"), fig2; px_per_unit = 3)
println("Saved phase_decades.png")
println("Done — outputs in $OUTDIR")
