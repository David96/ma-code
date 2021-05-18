using Distributed, ClusterManagers

addprocs(SlurmManager(20), partition="maxwell", t="01:30:00", nodes="6")

@everywhere begin
    include("BoostFactorOptimizer.jl")
    include("FileUtils.jl")

    using BoostFractor
    using Dates
    using DelimitedFiles

    freq_width= 50e6

    epsilon = 24
    n_disk = 20
    n_region = 2 * n_disk + 2
end

# %%


@sync @distributed for freq_center in (22e9 - 2.5e9):50e6:(22e9 + 2.5e9)
    init_spacing = 0.0
    if isfile("results/init_$(freq_center).txt")
        init_spacing = read_init_spacing_from_file("results/init_$freq_center.txt")
        println("Loaded init spacing from file: $init_spacing")
    end

    freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)
    eps = Array{Complex{Float64}}([i==1 ? 1e20 : i%2==0 ? 1 : epsilon for i=1:n_region])
    distance = distances_from_spacing(init_spacing, n_region)
    optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, freq_width, freq_range,
                                  distance, eps)

    if init_spacing == 0
        let
            best_spacing = 0.0
            best_boost = 0.0
            for i in 0.001:0.000001:0.01
                optim_params.sbdry_init.distance = distances_from_spacing(i, n_region)

                boost_factor = abs2(transformer(optim_params.sbdry_init, optim_params.coords,
                                                 optim_params.modes, prop=propagator1D,
                                                 reflect=nothing, f=freq_center,
                                                 diskR=optim_params.diskR)[1])
                if boost_factor > best_boost
                    best_boost = boost_factor
                    best_spacing = i
                end
            end
            println("Found resonant config at $best_spacing")
            write_init_spacing_to_file(best_spacing, freq_center)
            update_distances(optim_params, distances_from_spacing(best_spacing, n_region))
            init_spacing = best_spacing
        end
    end

    # %%

    spacings = optimize_spacings(optim_params, 0)

    # %%
    println("Writing spacings to file: $(spacings .+ init_spacing)")
    # init_spacing has to be added here because spacings are relative to it and we don't wanna care
    # about that when loading the spacings
    write_optim_spacing_to_file(spacings .+ init_spacing, freq_center)
end
