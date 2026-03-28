using JLD2

# ═══════════════════════════════════════════════════════════════════════
println("=== UDE+SINDy (noise=0.0) ===")
d = JLD2.load("results/ude_nonstationary_noise_0.0.jld2")

println("Selected λ = ", d["pysindy_selected_lambda"])
println("Feature names: ", d["pysindy_feature_names"])
println("f1 coefficients: ", round.(d["pysindy_coeffs_f1"], sigdigits=5))
println("f2 coefficients: ", round.(d["pysindy_coeffs_f2"], sigdigits=5))
println("Active f1: ", d["pysindy_active_f1"])
println("Active f2: ", d["pysindy_active_f2"])

function format_active(active, prefix)
    s = prefix
    for (name, c) in active
        s *= c >= 0 ? " + " : " - "
        s *= "$(round(abs(c), sigdigits=5))*$(name)"
    end
    return s
end

println("Full system:")
println("  dv/dt = ", format_active(d["pysindy_active_f1"], "0.05*v"))
println("  du/dt = ", format_active(d["pysindy_active_f2"], "-0.05*u"))

# ═══════════════════════════════════════════════════════════════════════
println("\n=== Direct SINDy (noise=0.0, seed=1) ===")
dd = JLD2.load("results/direct_sr3_nonstationary/results.jld2")
r = dd["results_all"][(0.0, 1)]

println("Feature names: ", r["feat_names"])
println("f1 coefficients: ", round.(r["coeffs_f1"], sigdigits=5))
println("f2 coefficients: ", round.(r["coeffs_f2"], sigdigits=5))
println("Active f1: ", r["active_f1"])
println("Active f2: ", r["active_f2"])

function format_direct(active)
    isempty(active) && return "0"
    parts = String[]
    for (i, (name, c)) in enumerate(active)
        sign = c >= 0 ? (i == 1 ? "" : " + ") : (i == 1 ? "-" : " - ")
        push!(parts, "$(sign)$(round(abs(c), sigdigits=5))*$(name)")
    end
    return join(parts)
end

println("Discovered equations:")
println("  dv/dt = ", format_direct(r["active_f1"]))
println("  du/dt = ", format_direct(r["active_f2"]))
