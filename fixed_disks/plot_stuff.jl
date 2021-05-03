include("FileUtils.jl")
include("BoostFactorOptimizer.jl")

using .BoostFactorOptimizer
using PyPlot

#freq = ARGS[1]
#shift = ARGS[2]
#version = ARGS[3]

# %%

freq = "2.2e10"
shift = "5.0e7"
version = "2021-04-26"

println("Plotting for $freq with shift $shift, v$version")

if shift[1] != '+' && shift[1] != '-'
    shift = "+$shift"
end

n_disk = 20
n_region = 2 * n_disk + 2
epsilon = 24

freq_center = parse(Float64, freq)
freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)

eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
init_spacings = read_optim_spacing_from_file("results/optim_$(freq)_$version.txt")

distances = distances_from_spacing(init_spacings)

optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, 0, freq_range,
                              distances, eps)

# %%
function plot_boostfactors()
    eout_0 = calc_eout(optim_params, zeros(n_disk))[1, 1, :]
    plot(freq_range .* 1e-9, abs2.(eout_0))

    for i in 1:n_disk
        optim_spacings = read_optim_spacing_from_file("results/optim_$freq$(shift)_f$(i)_$version.txt")
        plot(freq_range .* 1e-9,
             abs2.(calc_eout(optim_params, optim_spacings, fixed_disk = i)[1, 1, :]))
    end
end

function plot_boostfactor(shift, fixed_disk, ax)
    p = deepcopy(optim_params)
    p.freq_range = (freq_center - 100e6):0.004e9:(freq_center + shift + 100e6)
    eout_0 = calc_eout(p, zeros(n_disk))[1, 1, :]
    ax.plot(p.freq_range .* 1e-9, abs2.(eout_0))

    optim_spacings = read_optim_spacing_from_file(
                        "results/optim_$freq+$(shift)_f$(fixed_disk)_$version.txt")
    ax.plot(p.freq_range .* 1e-9,
         abs2.(calc_eout(p, optim_spacings, fixed_disk = fixed_disk)[1, 1, :]))

    ax.set_ylabel("Boostfactor")
    ax.set_xlabel("Frequency [GHz]")
end

function plot_disk_positions(fixed_disk::Int, shift)
    fig = figure()
    gs = fig.add_gridspec(2, height_ratios=[3, 1])
    axes = gs.subplots()
    fig.set_size_inches(10, 10)
    optim_spacings = read_optim_spacing_from_file(
                        "results/optim_$freq+$(shift)_f$(fixed_disk)_$version.txt")
    if fixed_disk > 0
        optim_spacings = vcat(optim_spacings[1:fixed_disk - 1],
                              -sum(optim_spacings[1:fixed_disk - 1]),
                              optim_spacings[fixed_disk:end])
    end
    #disk_rel_pos += optim_spacings
    plot_boostfactor(shift, fixed_disk, axes[1])
    plot_disk_positions(optim_spacings, axes[2])
    #savefig("mega_plot.png", dpi=400)
end

function plot_disk_positions(disk_rel_pos::Vector{Float64}, ax)
    # absolute positions have to take disk thickness (1mm) into account
    # abs_pos = [sum(disk_rel_pos[1:i]) + (i - 1) * 1e-3 for i in 1:length(disk_rel_pos)]
    real_rel_positions = [s + sum(disk_rel_pos[1:i - 1]) for (i, s) in enumerate(disk_rel_pos)]
    ax.set_xticks(1:length(disk_rel_pos))
    ax.bar(1:length(disk_rel_pos), real_rel_positions .* 1e3)
    ax.plot([0.4, length(disk_rel_pos) + 0.4], [0, 0])
    ax.set_ylabel("Disk shift [mm]")
    ax.set_xlabel("Disk")
end

function plot_bf_quality(fixed_disks, shift_range)
    qualities = Vector{Vector}(undef, length(fixed_disks))
    legend_text = Vector{String}(undef, length(fixed_disks))
    for (i, fd) in enumerate(fixed_disks)
        qualities[i] = Vector{Float64}()
        legend_text[i] = "Fixed disk $fd"
    end
    for shift in shift_range
        update_freq_center(optim_params, parse(Float64, freq) + shift)
        for (i, fixed_disk) in enumerate(fixed_disks)
            if shift == 0
                push!(qualities[i], 1.)
            else
                optim_spacings = read_optim_spacing_from_file("results/optim_$freq+$(shift)_f$(fixed_disk)_$version.txt")
                optim_spacings_0 = read_optim_spacing_from_file("results/optim_$freq+$(shift)_f0_$version.txt")
                cost_0 = calc_real_bf_cost(optim_params, optim_spacings_0)
                cost = calc_real_bf_cost(optim_params, optim_spacings, fixed_disk = fixed_disk)
                push!(qualities[i], cost / cost_0)
            end
        end
    end
    for quality in qualities
        plot(shift_range .* 1e-6, quality)
    end
    legend(legend_text, bbox_to_anchor=(1.01, 1.01))
    xlabel("Frequency shift [MHz]")
    ylabel("Boostfactor quality")
    ylim(0.9, 1.01)
end

#show()
