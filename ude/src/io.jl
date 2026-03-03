# src/io.jl
# Minimal I/O helpers for experiments (results folders + CSV saving)

export ensure_dir, make_run_dir, write_csv, write_vec_csv

using Dates
using DelimitedFiles

"Create directory if it doesn't exist. Returns the path."
function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
Create a timestamped run directory under results/<expname>/.
Example: results/exp_noise_length/2026-03-03_13-05-22/
Returns the full directory path.
"""
function make_run_dir(expname::AbstractString; root::AbstractString="results")
    ts = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    outdir = joinpath(root, expname, ts)
    ensure_dir(outdir)
    return outdir
end

"Write a matrix (or table-like array) to CSV."
function write_csv(path::AbstractString, A; delim=',')
    ensure_dir(dirname(path))
    writedlm(path, A, delim)
    return path
end

"Write a vector to a 1-column CSV."
function write_vec_csv(path::AbstractString, v; delim=',')
    ensure_dir(dirname(path))
    writedlm(path, v, delim)
    return path
end

export append_csv_row

function append_csv_row(path::AbstractString, A; delim=',')
    ensure_dir(dirname(path))
    open(path, "a") do io
        writedlm(io, A, delim)
    end
    return path
end