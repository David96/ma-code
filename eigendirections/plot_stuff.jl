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
    println(cost_0)
    plot([1, 20], [cost_0, cost_0])

    l = ["Reference"]
    x = Vector{Int}()
    y = Vector{Float64}()
    yerr = Vector{Float64}()
    for json in glob("spacings_*.json", dir)
        data = read_json(json)
        push!(x, data[1]["n_dim"])
        costs = Vector{Float64}()
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
        end
        push!(y, mean(costs))
        push!(yerr, std(costs))
        println("")
    end
    if length(x) > 0
        push!(l, "TODO")
        p = sortperm(x)
        x = x[p]
        y = y[p]
        yerr = yerr[p]
        errorbar(x, y, yerr=yerr)
    end
    xlabel("Dimensions")
    ylabel("Boostfactor magnitude")
    xticks(1:20)
    grid()
    legend(l)
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

