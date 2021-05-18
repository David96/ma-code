using Distributed, ClusterManagers

addprocs(SlurmManager(21), partition="maxwell", t="01:30:00", nodes="2-6")

@everywhere begin
    include("BoostFactorOptimizer.jl")
    include("FileUtils.jl")

    using BoostFractor

    # %%

    freq_center = 22e9
    freq_width = 50e6
    freq_shift = 50e6

    epsilon = 24
    n_disk = 20
    n_region = 2 * n_disk + 2

    freq_0_spacing = read_optim_spacing_from_file("results/optim_2.2e10_2021-04-26.txt")

    println("Loaded spacings from file: $freq_0_spacing")

    freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)
    eps = Array{Complex{Float64}}([i==1 ? 1e20 : i%2==0 ? 1 : epsilon for i=1:n_region])
    distance = distances_from_spacing(freq_0_spacing)
    optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, freq_width,
                                  freq_range, distance, eps)
    #eout_0 = calc_eout(optim_params, zeros(n_disk))[1, 1, :]
    #
    #plot(freq_range .* 1e-9, abs2.(eout_0))
    min_shift = -1500e6
end

# %%

#for i in 0:n_disk
#    spacings[i + 1] = read_optim_spacing_from_file("results/optim_2.2e10+1.1e9_f$(i)_2021-04-26.txt")
#end

@everywhere function optimize_for(p, fd, shift, s0)
    p = deepcopy(p)
    update_freq_center(p, freq_center + shift)
    spacing = @time optimize_spacings(p, fd, starting_point=s0)
    s = shift > 0 ? "+$(shift)" : "$(shift)"
    write_optim_spacing_to_file(spacing, "$freq_center$(s)_f$fd")
end

shift = parse(Float64, ARGS[1])
@sync for fd in 0:n_disk
    if abs(shift) != 50e6
        shift_before = shift < 0 ? "$(shift + 50e6)" : "+$(shift - 50e6)"
        s0 = read_optim_spacing_from_file("results/optim_$(freq_center)$(shift_before)_f$(fd)_v1.txt")
    else
        s0 = zeros(n_disk - (fd == 0 ? 0 : 1))
    end
    @spawnat (fd + 2) optimize_for(optim_params, fd, shift, s0)
    println("Spawned worker $(fd)…")
end

println("All good, we done")

# %%

#clf()
#legend_text = ["freq_0"]
#plot(freq_range .* 1e-9, abs2.(eout_0))
#
#for i in 1:n_disk
#    push!(legend_text, "Fixed disk: $i")
#    plot(freq_range .* 1e-9, abs2.(eout[i]))
#end
#
#xlabel("Frequency [GHz]")
#ylabel("Power Boostfactor")
#legend(legend_text)
