# experiments/viz_deriv_comparison.jl
#
# Two-panel figure: derivative RMSE and R² vs noise level,
# SINDy (mean ± std band) vs UDE+SINDy.
#
# Loads summary_combined.csv from the latest results/deriv_comparison/ dir.
# Saves deriv_comparison.png to the same directory.
#
# Usage:  julia --project=. experiments/viz_deriv_comparison.jl

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

using Plots, DelimitedFiles, Statistics
gr()

# ─── Locate latest results directory ─────────────────────────────────────────

function find_latest_rundir(expname; root = "results")
    base = joinpath(root, expname)
    isdir(base) || error("No results directory: $base")
    entries = filter(isdir, readdir(base; join = true))
    isempty(entries) && error("No run directories in: $base")
    return last(sort(entries))
end

results_root = joinpath(@__DIR__, "..", "results")
outdir = if length(ARGS) >= 1
    ARGS[1]
else
    find_latest_rundir("deriv_comparison"; root = results_root)
end

csv_path = joinpath(outdir, "summary_combined.csv")
isfile(csv_path) || error("Not found: $csv_path")
println("Loading: ", csv_path)

# ─── Parse CSV ───────────────────────────────────────────────────────────────

data, hdr = readdlm(csv_path, ',', header = true)
hdr = vec(hdr)

col(name) = findfirst(==(name), hdr)

ci_method = col("method")
ci_noise  = col("noise")
ci_rmse   = col("global_RMSE")
ci_r2     = col("global_R2")

function parse_or_nan(v)
    v isa Number && return Float64(v)
    s = strip(string(v))
    (s == "NaN" || s == "failed" || isempty(s)) && return NaN
    return parse(Float64, s)
end

noise_levels = [0.0, 1e-3, 1e-2, 5e-2]
n_noise      = length(noise_levels)

# ─── Aggregate SINDy (mean ± std across seeds) ──────────────────────────────

sindy_rows = findall(r -> data[r, ci_method] == "sindy", axes(data, 1))
ude_rows   = findall(r -> data[r, ci_method] == "ude_sindy", axes(data, 1))

sindy_rmse_mean = fill(NaN, n_noise)
sindy_rmse_std  = fill(NaN, n_noise)
sindy_r2_mean   = fill(NaN, n_noise)
sindy_r2_std    = fill(NaN, n_noise)

ude_rmse = fill(NaN, n_noise)
ude_r2   = fill(NaN, n_noise)

for (ni, nv) in enumerate(noise_levels)
    # SINDy: collect valid seeds
    rmses = Float64[]
    r2s   = Float64[]
    for r in sindy_rows
        abs(parse_or_nan(data[r, ci_noise]) - nv) > 1e-12 && continue
        rv = parse_or_nan(data[r, ci_rmse])
        isnan(rv) && continue
        push!(rmses, rv)
        push!(r2s, parse_or_nan(data[r, ci_r2]))
    end
    if !isempty(rmses)
        sindy_rmse_mean[ni] = mean(rmses)
        sindy_rmse_std[ni]  = length(rmses) > 1 ? std(rmses) : 0.0
        sindy_r2_mean[ni]   = mean(r2s)
        sindy_r2_std[ni]    = length(r2s) > 1 ? std(r2s) : 0.0
    end

    # UDE+SINDy: single value
    for r in ude_rows
        abs(parse_or_nan(data[r, ci_noise]) - nv) > 1e-12 && continue
        ude_rmse[ni] = parse_or_nan(data[r, ci_rmse])
        ude_r2[ni]   = parse_or_nan(data[r, ci_r2])
    end
end

# ─── Plotting (paper-ready) ──────────────────────────────────────────────────

noise_labels = ["0", "10⁻³", "10⁻²", "5×10⁻²"]
xs = 1:n_noise   # categorical positions

FONT_TICK  = 11
FONT_LABEL = 13
FONT_ANNOT = 12

# ── Panel (a): derivative RMSE ───────────────────────────────────────────────

p_rmse = plot(;
    ylabel       = "Derivative RMSE",
    xlabel       = "Noise level (σ)",
    legend       = false,
    xticks       = (xs, noise_labels),
    tickfontsize = FONT_TICK,
    guidefontsize = FONT_LABEL,
    leftmargin   = 4Plots.mm,
    bottommargin = 6Plots.mm,
    topmargin    = 2Plots.mm,
    rightmargin  = 2Plots.mm,
)

plot!(p_rmse, xs, sindy_rmse_mean;
      ribbon     = sindy_rmse_std,
      fillalpha  = 0.2,
      color      = :steelblue,
      linewidth  = 2.0,
      marker     = :circle,
      markersize = 5,
      label      = "")

plot!(p_rmse, xs, ude_rmse;
      color      = :firebrick,
      linewidth  = 2.0,
      linestyle  = :dash,
      marker     = :diamond,
      markersize = 5,
      label      = "")

# Panel label
annotate!(p_rmse, 1.1, minimum(filter(!isnan, [sindy_rmse_mean .- sindy_rmse_std; ude_rmse])) * 0.97,
          text("(a)", FONT_ANNOT, :left, :bottom))

# ── Panel (b): derivative R² ─────────────────────────────────────────────────

p_r2 = plot(;
    ylabel        = "Derivative R²",
    xlabel        = "Noise level (σ)",
    legend        = :topright,
    legendfontsize = FONT_TICK,
    xticks        = (xs, noise_labels),
    tickfontsize  = FONT_TICK,
    guidefontsize = FONT_LABEL,
    leftmargin    = 4Plots.mm,
    bottommargin  = 6Plots.mm,
    topmargin     = 2Plots.mm,
    rightmargin   = 3Plots.mm,
)

plot!(p_r2, xs, sindy_r2_mean;
      ribbon     = sindy_r2_std,
      fillalpha  = 0.2,
      color      = :steelblue,
      linewidth  = 2.0,
      marker     = :circle,
      markersize = 5,
      label      = "SINDy (mean ± std)")

plot!(p_r2, xs, ude_r2;
      color      = :firebrick,
      linewidth  = 2.0,
      linestyle  = :dash,
      marker     = :diamond,
      markersize = 5,
      label      = "UDE+SINDy")

# Panel label
y_lo_r2 = minimum(filter(!isnan, [sindy_r2_mean .- sindy_r2_std; ude_r2]))
annotate!(p_r2, 1.1, y_lo_r2 - 0.02 * abs(y_lo_r2),
          text("(b)", FONT_ANNOT, :left, :bottom))

# ── Combine into figure ──────────────────────────────────────────────────────

p = plot(p_rmse, p_r2;
         layout = (1, 2),
         size   = (820, 370),
         dpi    = 300,
)

outpath = joinpath(outdir, "deriv_comparison.png")
savefig(p, outpath)
println("Saved: ", outpath)
