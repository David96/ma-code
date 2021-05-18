include("FileUtils.jl")
include("BoostFactorOptimizer.jl")

using BoostFractor
using PyPlot

#freq = ARGS[1]
#shift = ARGS[2]
#version = ARGS[3]

# %%

freq = "2.2e10"
shift = "5.0e7"
version = "v1"

println("Plotting for $freq with shift $shift, $version")

if shift[1] != '+' && shift[1] != '-'
    shift = "+$shift"
end

n_disk = 20
n_region = 2 * n_disk + 2
epsilon = 24

freq_center = parse(Float64, freq)
freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)

eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
init_spacings = read_optim_spacing_from_file("results/optim_$(freq)_2021-04-26.txt")

distances = distances_from_spacing(init_spacings)

optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, 50e6, freq_range,
                              distances, eps)

# %%
shift_text(shift) = shift < 0 ? "$(shift)" : "+$(shift)"

function plot_boostfactors()
    eout_0 = calc_eout(optim_params, zeros(n_disk))[1, 1, :]
    plot(freq_range .* 1e-9, abs2.(eout_0))

    for i in 1:n_disk
        optim_spacings = read_optim_spacing_from_file(
                          "results/optim_$freq$(shift_text(shift))_f$(i)_$version.txt")
        plot(freq_range .* 1e-9,
             abs2.(calc_eout(optim_params, optim_spacings, fixed_disk = i)[1, 1, :]))
    end
end

function plot_boostfactor(shift, fixed_disk, ax)
    p = deepcopy(optim_params)
    if shift < 0
        p.freq_range = (freq_center + shift - 100e6):0.004e9:(freq_center + 100e6)
    else
        p.freq_range = (freq_center - 100e6):0.004e9:(freq_center + shift + 100e6)
    end
    eout_0 = calc_eout(p, zeros(n_disk))[1, 1, :]
    ax.plot(p.freq_range .* 1e-9, abs2.(eout_0))

    optim_spacings = read_optim_spacing_from_file(
                      "results/optim_$freq$(shift_text(shift))_f$(fixed_disk)_$version.txt")
    if fixed_disk > 0
        optim_spacings_0 = read_optim_spacing_from_file(
                            "results/optim_$(parse(Float64, freq) + shift)_v1.txt")
        update_distances(p, distances_from_spacing(optim_spacings_0),
                         update_itp = false)
        ax.plot(p.freq_range .* 1e-9, abs2.(calc_eout(p, zeros(n_disk))[1, 1, :]))
        update_distances(p, distances, update_itp=false)
    end
    ax.plot(p.freq_range .* 1e-9,
         abs2.(calc_eout(p, optim_spacings, fixed_disk = fixed_disk)[1, 1, :]))
    ax.legend(["\$f_0\$", "Free", "Fixed disk $fixed_disk"])

    ax.set_ylabel("Boostfactor")
    ax.set_xlabel("Frequency [GHz]")
end

function plot_disk_positions(fixed_disk::Int, shift; christoph=false)
    fig = figure()
    #fig.suptitle("Fixed disk $fixed_disk, shift $(shift * 1e-6) MHz", fontsize=16)
    gs = fig.add_gridspec(2, height_ratios=[3, 1])
    axes = gs.subplots()
    fig.set_size_inches(10, 10)
    optim_spacings = read_optim_spacing_from_file(
                      "results/optim_$freq$(shift_text(shift))_f$(fixed_disk)_$version.txt")
    if fixed_disk > 0
        optim_spacings = vcat(optim_spacings[1:fixed_disk - 1],
                              -sum(optim_spacings[1:fixed_disk - 1]),
                              optim_spacings[fixed_disk:end])
    end
    #disk_rel_pos += optim_spacings
    plot_boostfactor(shift, fixed_disk, axes[1])
    if christoph
        optim_spacings_0 = read_optim_spacing_from_file(
                            "results/optim_$freq$(shift_text(shift))_f0_$version.txt")
        plot_disk_positions(optim_spacings - optim_spacings_0, axes[2])
    else
        plot_disk_positions(optim_spacings, axes[2])
    end
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

function plot_bf_quality(fixed_disks, shift_range; area=true)
    qualities = Vector{Vector}(undef, length(fixed_disks))
    legend_text = Vector{String}(undef, length(fixed_disks))
    for (i, fd) in enumerate(fixed_disks)
        qualities[i] = Vector{Float64}()
        legend_text[i] = "Fixed disk $fd"
    end
    for shift in shift_range
        update_freq_center(optim_params, parse(Float64, freq) + shift, update_itp = false)
        optim_spacings_0 = read_optim_spacing_from_file(
                            "results/optim_$(parse(Float64, freq) + shift)_v1.txt")
        update_distances(optim_params, distances_from_spacing(optim_spacings_0),
                         update_itp = false)
        cost_0 = calc_real_bf_cost(optim_params, zeros(n_disk), fixed_disk=0, area=area)
        update_distances(optim_params, distances, update_itp=false)
        for (i, fixed_disk) in enumerate(fixed_disks)
            if shift == 0
                push!(qualities[i], 1.)
            else
                s = shift_text(shift)
                optim_spacings = read_optim_spacing_from_file(
                                    "results/optim_$freq$(s)_f$(fixed_disk)_$version.txt")
                try
                    cost = calc_real_bf_cost(optim_params, optim_spacings, fixed_disk = fixed_disk,
                                             area=area)
                    push!(qualities[i], cost / cost_0)
                catch e
                    println("Shift $s, fd $fixed_disk failed: ", e)
                end
            end
        end
    end
    figure().set_size_inches(10, 8)
    ax = subplot(1, 1, 1)
    #println(qualities)
    for quality in qualities
        ax.plot(shift_range .* 1e-9, quality)
    end
    ax.legend(legend_text, bbox_to_anchor=(1.01, 1.01))
    ax.set_xlabel("Frequency shift [GHz]")
    ax.set_ylabel("Boostfactor quality")
    ax.set_xticks(-2.5:0.5:2.5)
    ax.set_xticks(-2.5:0.05:2.5, minor=true)
    #grid(xdata=shift_range .* 1e-6, ydata=0.65:0.05:1.15)
    ax.grid(which="minor", alpha=0.2)
    ax.grid(which="major", alpha=0.5)
    ax.set_ylim(0.65, 1.05)
end

function plot_trace_back(fixed_disk, freq, shift)
    optim_spacings = read_optim_spacing_from_file("results/optim_$(freq)$(shift_text(shift))_f$(fixed_disk)_$version.txt")
    if fixed_disk != 0
        optim_spacings = spacings_with_fd(optim_spacings, fixed_disk)
    end
    bdry = copy_setup_boundaries(optim_params.sbdry_init, optim_params.coords)
    bdry.distance[2:2:end-2] .+= optim_spacings
    p = deepcopy(optim_params)
    p.freq_range = freq
    # set fixed_disk to 0 here as we already inserted the missing distance
    refl = calc_eout(p, optim_spacings, fixed_disk=0, reflect=true)[1][1]
    full_fields = BoostFractor.transformer_trace_back(refl, p.m_reflect, bdry, p.coords, p.modes,
                                         prop=propagator1D, f=freq)
    figure().set_size_inches(15, 12)
    plot_1d_field_pattern(-autorotate(full_fields[:,:,1]), bdry, freq)
end

"""
    Plot the fields inside the system for a 1D calculation
"""
function plot_1d_field_pattern(full_solution_regions, bdry::SetupBoundaries, f; fill=false,
    add_ea=false, overallphase=1)
    # Iterate over regions and plot the result in each region
    ztot = 0 # Each region needs to know where it starts, so iteratively add up the lengths of regions
    Nregions = length(bdry.eps)
    for s in 1:Nregions
        # Propagation constant in that region
        c=299792458
        kreg = 2pi/c*f
        kreg *= sqrt(bdry.eps[s])

        # Define the color of that region according to mirror / disk / free space / ...
        fillcolor = nothing
        if abs.(bdry.eps[s]) > 100 || bdry.eps[s] == NaN
            fillcolor = "darkorange"
        elseif abs.(bdry.eps[s]) != 1
            fillcolor = "lightgray"
        end

        # Plot that region
        plot_region_1d(full_solution_regions[s,1].*overallphase, full_solution_regions[s,2].*overallphase,
                        ztot, ztot+bdry.distance[s],
                        kreg,
                        Ea=(add_ea ? (1/bdry.eps[s]).*overallphase : 0),
                        maxE=2.2*maximum(abs.(full_solution_regions[:,:])),
                        bgcolor=fillcolor, extraspace=(s == Nregions),fill=fill,)

        ztot += bdry.distance[s]
    end

    # Add annotations to plot
    xlabel("z [m]")
    ylabel("\$E/E_0\$")

    legend(loc="lower right")
end

"""
    Plot the fields inside one region for a 1D calculation
"""
function plot_region_1d(R, L, z0, z1, k; bgcolor=nothing, extraspace=false,fill=false, Ea=0,maxE=10)    
    # Construct the relative coordinate system for that region
    z1 += 1e-9
    maximum = (z1+(extraspace ? 10e-3 : 0))
    z = vcat(z0:2e-4:maximum, maximum)
    dz = z .- z0
    dz2 = .-(z .- z1)

    # Calculate the functional solution for the region
    Rf = L*exp.(+1im.*k.*dz2)
    Lf = R*exp.(-1im.*k.*dz2)

    #Plot

    # Mark the Region as Disk / Mirror / Air / etc.
    if bgcolor !== nothing
        fill_between([z0 == 0 ? -0.0025 : z0,z1], -maxE, maxE, color=bgcolor, linewidth=0)
    end

    # Plot the real and imaginary part of the solution
    plot(z, real.(Rf.+Lf.+ Ea), c="b", label=(extraspace ? "Re(E)" : ""))
    plot(z, imag.(Rf.+Lf.+ Ea), c="r", label=(extraspace ? "Im(E)" : ""))
end

"""
    Rotate the solution such that the fields inside are real
"""
function autorotate(full_solution_regions)
    ang = angle.(full_solution_regions[2,1].+full_solution_regions[2,2])
    #sgn = real.((full_solution_regions[2,1].+full_solution_regions[2,2]).*exp(-1im*ang)) .> 0
    return full_solution_regions.*exp(-1im*ang)
end
#show()
