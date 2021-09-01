using Distributed, ClusterManagers, Glob

if haskey(ENV, "SLURM_AVAILABLE") && ENV["SLURM_AVAILABLE"] == "true"
    addprocs(SlurmManager(50), partition="maxwell", t="02:00:00", nodes="2-6", kill_on_bad_exit="1",
             cpus_per_task="8")
end

@everywhere begin
    using BoostFractor
    using Optim
    using ForwardDiff
    using LinearAlgebra
    using Statistics

    include("../common/BoostFactorOptimizer.jl")
    include("../common/FileUtils.jl")

    n_disk = 20
    n_region = 2 * n_disk + 2
    epsilon = 24

    #Gradient of cost function
    function grad_cost_fun(x, p::BoosterParams)
        #ForwardDiff.gradient(cost_fun(p, 0), x)
        cost_fun(p, 0, gradient=true)(x)
    end

    function contains_nan_or_inf(C)
        for c_ij in C
            if isnan(c_ij) || isinf(c_ij)
                return true
            end
        end
        return false
    end

    # Calculates C matrix via Monte Carlos Sampling of cost function gradient around 
    # disk spacing = initial_spacing + x_0
    # M = Number of samples.
    # Takes a lot to converge but this might not be important as different eigendirections
    # seem to work equally well
    function calc_C_matrix(x_0, p::BoosterParams; M=1000, variation = 100e-6)
        while true
            C_matrix = zeros(n_disk, n_disk)

            for i=1:M
                x_i = x_0 .+ 2 .* (rand(n_disk).-0.5) .* variation
                _, grad = grad_cost_fun(x_i, p)
                grad = grad[1:n_disk] # remove the last entry corresponding to the gap between last disk and antenna
                C_matrix += (grad / M) .* transpose(grad)
            end
            if !contains_nan_or_inf(C_matrix)
                return C_matrix
            end
        end
    end

    function get_init_spacings(optim_params::BoosterParams)
        best_spacing = 0.0
        best_boost = 0.0
        for i in 0.005:0.000001:0.015
            optim_params.sbdry_init.distance = distances_from_spacing(i, n_region)

            boost_factor = abs2(transformer(optim_params.sbdry_init, optim_params.coords,
                                            optim_params.modes, prop=propagator1D,
                                            reflect=nothing, f=optim_params.freq_center,
                                            diskR=optim_params.diskR)[1])
            if boost_factor > best_boost
                best_boost = boost_factor
                best_spacing = i
            end
        end
        return best_spacing
    end

    function get_optim_params(freq; freq_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9),
                              update_itp=true)
        eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
        init_spacing = 0.0
        distances = distances_from_spacing(init_spacing, n_region)
        optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq, 50e6, freq_range,
                                      distances, eps, update_itp=update_itp)
        if isfile("results/init_$(freq).txt")
            init_spacing = read_init_spacing_from_file("results/init_$(freq).txt")
        else
            init_spacing = get_init_spacings(optim_params)
            write_init_spacing_to_file(init_spacing, freq)
        end
        update_distances(optim_params, distances_from_spacing(init_spacing, n_region))
        return optim_params
    end

    function calc_eigendirections(freq; M=2000, variation = 60e-6)
        optim_params = get_optim_params(freq)

        #Eigenvalue decomposition
        C_matrix = calc_C_matrix(zeros(n_disk), optim_params, M=M, variation=variation)
        eigen_decomp = eigen(C_matrix)
        eigenvalues = eigen_decomp.values
        eigendirections = eigen_decomp.vectors

        #Sort by eigenvalues
        p = sortperm(eigenvalues,rev=true)
        eigenvalues = eigenvalues[p]
        eigendirections = eigendirections[:,p]
        return eigenvalues, eigendirections
    end

    function optimize_bf_with_eigendirections(freq; M=2000, variation=60e-6, n=1024, n_dim=5,
            eigendirections=nothing)
        if eigendirections === nothing && n_dim > 0
            eigenvalues, eigendirections = calc_eigendirections(freq, M=M, variation=variation)
        end
        optim_params = get_optim_params(freq)
        if n_dim == 0
            time = @elapsed optim_spacings =
                                optimize_spacings(optim_params, 0, n=n)
        else
            time = @elapsed optim_spacings =
                                optimize_spacings(optim_params, 0, n=n, starting_point=zeros(n_dim),
                                            cost_function=cost_fun_rot(optim_params, eigendirections))
            # optim_spacings = eigendirections[:,1:length(optim_spacings)] * optim_spacings
        end
        time, optim_spacings
    end

    function save_spacing_to_json(filename, freq, M, variation, n_dim, n, time, spacing)
        data = Dict(:freq => freq, :M => M, :variation => variation, :n_dim => n_dim, :n => n,
                    :time => time, :spacing => spacing)
        write_json("results/$filename", data)
    end
end

function find_convergence(freq, M_range; N=50, variation=100e-6)
    diffs_var = @distributed (vcat) for M in M_range[1:end]
        println("M=$(M)…")
        diff = Vector{Float64}()
        last_ev, _ = calc_eigendirections(freq, M=M, variation=variation)
        for i = 1:N
            cur_ev, _ = calc_eigendirections(freq, M=M, variation=variation)
            push!(diff, sum(abs.(cur_ev / sum(cur_ev) - last_ev / sum(last_ev))))
            last_ev = cur_ev
        end
        mean_diff = mean(diff)
        variance = var(diff)
        println("Diff = $(mean_diff)\nVariance: $variance")
        mean_diff, variance
    end
    data = Dict(:M_range => collect(M_range), :N => N, :variation => variation, :freq => freq,
                :mean => first.(diffs_var), :variance => map(x -> x[2], diffs_var))
    write_json("diffs_v=$variation.json", data)
    #title("Eigenvalue differences at $freq")
    #ylabel("Difference")
    #xlabel("M")
    #errorbar(M_range[1:end], first.(diffs_var), yerr=map(x -> x[2], diffs_var))
    #savefig("ev_diffs.png")
end

function optimize_freq_range(freq_range; M=2000, variation=60e-6, n=1024, n_dim=5)
    @sync @distributed for freq in freq_range
        time, optim_spacings = optimize_bf_with_eigendirections(freq, M=M, variation=variation,
                                                                n=n, n_dim=n_dim)
        save_spacing_to_json("$(freq)_$(M)_$(variation)_$(n_dim).json", freq, M, variation,
                             n_dim, n, time, optim_spacings)
    end
end

function optimize_dim_range(freq, dim_range; M=2000, variation=60e-6, n=1024, rep=100)
    _, eigendirections = calc_eigendirections(freq, M=M, variation=variation)
    for n_dim in dim_range
        reps = @distributed (vcat) for i in 1:rep
            time, optim_spacings = optimize_bf_with_eigendirections(freq, M=M, variation=variation,
                                                                    n=n, n_dim=n_dim,
                                                                    eigendirections=eigendirections)
            Dict(:freq => freq, :M => M, :variation => variation, :n_dim => n_dim, :n => n,
                        :time => time, :spacing => optim_spacings)
        end
        write_json("spacings_n=$(n)_n-dim=$(n_dim).json", reps)
    end
end

function scan_freq_range(start_freq, freq_range; M=2000, variation=60e-6, n=128, n_dim=3,
                         dir="results", ref=false)
    _, eigendirections = calc_eigendirections(start_freq, M=M, variation=variation)
    freqs = @distributed (vcat) for freq in freq_range
        if ref
            _, eigendirections = calc_eigendirections(freq, M=M, variation=variation)
        end
        time, optim_spacings = optimize_bf_with_eigendirections(freq,
                                                                eigendirections=eigendirections,
                                                                n_dim=n_dim, n=n)

        Dict(:freq => freq, :n_dim => n_dim, :n => n, :time => time, :spacing =>
             optim_spacings, :eigendirections => eigendirections, :ref => ref)
    end
    if ref
        write_json("$dir/spacings_ref_n=$(n)_n-dim=$(n_dim).json", freqs)
    else
        write_json("$dir/spacings_n=$(n)_n-dim=$(n_dim).json", freqs)
    end
end

#function update(plt, eigendirections, i)
#    return frame -> begin
#        for (j, b) in enumerate(plt)
#            b.set_height(eigendirections[frame][j, i])
#        end
#    end
#end
#
#function anim_eigendirection(i)
#    fig, ax = subplots()
#    eigendirections = Dict{Int, Array{Float64, 2}}()
#    for f = 12e9:1e9:29e9
#        _, eigendirections[f] = calc_eigendirections(f)
#    end
#    plt = bar(collect(1:20), eigendirections[12e9][:, i])
#    myanim = anim.FuncAnimation(fig, update(plt, eigendirections, i), frames=12e9:1e9:29e9)
#    myanim[:save]("test1.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
#end
