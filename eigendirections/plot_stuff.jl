include("../common/FileUtils.jl")
include("../common/BoostFactorOptimizer.jl")
include("eigendirections.jl")

using BoostFractor
using Glob
using PyPlot
using Statistics

n_disk = 20
n_region = 2 * n_disk + 2
epsilon = 24

eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)

function plot_dims(dir; area=true)
    freq_center = 22e9
    freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)
    init_spacings = read_optim_spacing_from_file("results/optim_2.2e10_v1.txt")
    distances = distances_from_spacing(init_spacings)
    optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, 50e6, freq_range,
                                  distances, eps)
    cost_0 = calc_real_bf_cost(optim_params, zeros(n_disk), fixed_disk=0, area=area)
    cost_0 = -convert(Float64, cost_0) / 50e6
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot([1, 20], [cost_0, cost_0])

    l = ["Reference"]
    x = Dict{Int, Vector{Int}}()
    y = Dict{Int, Vector{Float64}}()
    yerr = Dict{Int, Vector{Float64}}()
    time = Dict{Int, Vector{Float64}}()
    time_err = Dict{Int, Vector{Float64}}()
    for json in glob("spacings_*.json", dir)
        data = read_json(json)
        if !haskey(x, data[1]["n"])
            x[data[1]["n"]] = Vector{Int}()
            y[data[1]["n"]] = Vector{Float64}()
            yerr[data[1]["n"]] = Vector{Float64}()
            time[data[1]["n"]] = Vector{Float64}()
            time_err[data[1]["n"]] = Vector{Float64}()
        end
        x_n = x[data[1]["n"]]
        y_n = y[data[1]["n"]]
        yerr_n = yerr[data[1]["n"]]
        time_n = time[data[1]["n"]]
        time_err_n = time_err[data[1]["n"]]
        push!(x_n, data[1]["n_dim"])
        costs = Vector{Float64}()
        times = Vector{Float64}()
        init_spacing = read_init_spacing_from_file("results/init_2.2e10.txt")
        freq = data[1]["freq"]
        f_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9)
        distances = distances_from_spacing(init_spacing, n_region)
        p = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq, 50e6, f_range,
                                      distances, eps)
        for d in data
            spacings = convert(Vector{Float64}, d["spacing"])
            cost_0 = calc_real_bf_cost(p, spacings, fixed_disk=0, area=area)
            cost_0 = -cost_0 / 50e6
            push!(costs, cost_0)
            push!(times, d["time"])
        end
        push!(y_n, mean(costs))
        push!(yerr_n, std(costs))
        push!(time_n, mean(times))
        push!(time_err_n, std(times))
    end
    ax2.plot([], [])
    for (n, x_n) in x
        push!(l, "n=$n")
        p = sortperm(x_n)
        x_n = x_n[p]
        y_n = y[n][p]
        time_n = time[n][p]
        yerr_n = yerr[n][p]
        time_err_n = time_err[n][p]
        ax1.errorbar(x_n, y_n, yerr=yerr_n)
        ax2.errorbar(x_n, time_n, yerr=time_err_n, ls="-.")
    end
    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Boostfactor magnitude")
    ax2.set_ylabel("Time [s]")
    xticks(1:20)
    grid()
    ax1.legend(l, bbox_to_anchor=(1.1, 1.01))
end

function plot_diffs(dir)
    legend_text = []
    for json in glob("diffs_*.json", dir)
        println("Found file $json")
        data = read_json(json)
        x = data["M_range"]
        y = data["mean"]
        error = data["variance"]
        title("Eigenvalue differences at $(data["freq"])")
        ylabel("Difference")
        xlabel("M")
        errorbar(x, y, yerr=error)
        push!(legend_text, data["variation"])
    end
    legend(legend_text)
end

function plot_eigendirections(freq, n)
    eigenvalues, eigendirections = calc_eigendirections(freq)
    figure().set_size_inches(12, cld(n, 3) * 3)
    subplot(cld(n, 3), 3, 1)
    bar(collect(1:20), eigenvalues ./ sum(eigenvalues))
    xlabel("i")
    ylabel("\$\\lambda_i \$")
    yscale("log")
    title("Eigenvalues")

    for i = 1:n
        subplot(cld(n, 3), 3, i+1)
        bar(collect(1:20),eigendirections[:,i])
        xlabel("\$i\$")
        ylabel("\$\\d_i \$")
        ylim(-0.6, 0.6)
        title("Eigendirection $i")
    end

    tight_layout()
end

