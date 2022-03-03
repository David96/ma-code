using Distributed, ClusterManagers, Glob, HCubature

using BoostFractor
using Optim
using ForwardDiff
using LinearAlgebra
using Statistics
using LsqFit

epsilon = 24

#Gradient of cost function
function grad_cost_fun(x, p::BoosterParams; cost=nothing)
    if cost !== nothing
        ForwardDiff.gradient(cost, x)
    else
        #ForwardDiff.gradient(cost_fun(p, 0), x)
        _, grad = cost_fun(p, 0, gradient=true)(x)
        return grad[1:p.n_disk]
    end
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
function calc_C_matrix(x_0, p::BoosterParams; M=1000, variation = 50-6, cost_fun=nothing)
    if cost_fun === nothing
        f = x -> grad_cost_fun(x, p)
    else
        f = x -> grad_cost_fun(x, p, cost=cost_fun)
    end
    #integral, error = hcubature((x) -> f(x) * transpose(f(x)) / variation^p.n_disk, x_0 .- variation / 2,
    #                     x_0 .+ variation / 2, rtol = 0.05,
    #                     norm=a -> sum([x*x for x in a]))
    #display(integral)
    #display(sqrt(error))
    #return integral
    while true
        C_matrix = zeros(length(x_0), length(x_0))

        for _=1:M
            x_i = x_0 .+ 2 .* (rand(length(x_0)).-0.5) .* variation
            if cost_fun !== nothing
                grad = grad_cost_fun(x_i, p, cost=cost_fun)
            else
                grad = grad_cost_fun(x_i, p)
                #grad = grad[1:p.n_disk] # remove the last entry corresponding to the gap between last disk and antenna
            end
            C_matrix += (grad / M) .* transpose(grad)
        end
        if !contains_nan_or_inf(C_matrix)
            return C_matrix
        end
    end
end

function calc_eigendirections(freq, optim_params = get_optim_params(freq);
        M=2000, variation = 50e-6, cost_fun=nothing, n_params=optim_params.n_disk)
    #Eigenvalue decomposition
    C_matrix = calc_C_matrix(zeros(n_params), optim_params, M=M, variation=variation,
                             cost_fun=cost_fun)
    eigen_decomp = eigen(C_matrix)
    eigenvalues = eigen_decomp.values
    eigendirections = eigen_decomp.vectors

    #Sort by eigenvalues
    p = sortperm(eigenvalues,rev=true)
    eigenvalues = eigenvalues[p]
    eigendirections = eigendirections[:,p]
    return eigenvalues, eigendirections
end

oscillate(order, n, i) = sin(i * order * pi / n) * 2 * pi / n

function gen_eigendirection(order; n_disk=20)
    [oscillate(order, n_disk, i) for i in 1:n_disk]
end

function gen_eigendirections(; n_disk=20)
    hcat([gen_eigendirection(order) for order in 1:n_disk]...)
end

function optimize_bf_with_eigendirections(freq, optim_params = get_optim_params(freq);
        M=2000, variation=60e-6, n=1024, n_dim=5, eigendirections=nothing, 
        starting_point=zeros(n_dim),
        algorithm=BFGS(linesearch=BackTracking(order=2)),
        options=Optim.Options(f_tol=1e-6), kwargs...)
    if eigendirections === nothing && n_dim > 0
        eigenvalues, eigendirections = calc_eigendirections(freq, M=M, variation=variation,
                                                           cost_fun=cost_fun(optim_params,
                                                                             0; kwargs...))
    end
    if n_dim == 0
        time = @elapsed optim_spacings =
                            Optim.minimizer(optimize_spacings(optim_params, 0, n=n, algorithm=algorithm,
                                              options=options, starting_point=starting_point,
                                              cost_function=cost_fun(optim_params, 0;
                                                                     kwargs...)))
    else
        time = @elapsed optim_spacings =
                            Optim.minimizer(optimize_spacings(optim_params, 0, n=n, algorithm=algorithm,
                                              options=options, starting_point=starting_point,
                                              cost_function=cost_fun_rot(optim_params,
                                                                         eigendirections; kwargs...)))
        # optim_spacings = eigendirections[:,1:length(optim_spacings)] * optim_spacings
    end
    time, optim_spacings
end

function save_spacing_to_json(filename, freq, M, variation, n_dim, n, time, spacing)
    data = Dict(:freq => freq, :M => M, :variation => variation, :n_dim => n_dim, :n => n,
                :time => time, :spacing => spacing)
    write_json("results/$filename", data)
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
    for n_dim in dim_range
        reps = @distributed (vcat) for i in 1:rep
            _, eigendirections = calc_eigendirections(freq, M=M, variation=variation)
            time, optim_spacings = optimize_bf_with_eigendirections(freq, M=M, variation=variation,
                                                                    n=n, n_dim=n_dim,
                                                                    eigendirections=eigendirections)
            Dict(:freq => freq, :M => M, :variation => variation, :n_dim => n_dim, :n => n,
                        :time => time, :spacing => optim_spacings,
                        :eigendirections => eigendirections)
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

#@. fit_function(x, p) = p[1] * (x - p[2])^2 + p[3]

function shift_bf(start_freq; n_dim=3, dir="results_shift")
    freqs = Vector{Int}()
    spacings = Vector{Vector{Float64}}()
    _, ed_tmp= calc_eigendirections(start_freq, variation=50e-6)
    p = get_optim_params(22e9)
    _, spacing = optimize_bf_with_eigendirections(22e9, p, n_dim=n_dim,
                                                   eigendirections=ed_tmp, n=64)
    p.sbdry_init.distance[2:2:end-2] .+= ed_tmp[:, 1:n_dim] * spacing

    _, eigendirections = calc_eigendirections(22e9, p, cost_fun=cost_fun(p, 0))
#    n_dim = n_dim + 1
#    eigendirections[:, n_dim] = fill(1 / n_disk, n_disk)
#    init_spacing_0 = get_init_spacings(start_freq)
#    optim_params = get_optim_params(start_freq)
    for freq in (start_freq - 2e9):0.1e9:(start_freq + 2e9)
        optim_params = get_optim_params(freq)
        update_freq_center(optim_params, freq)
        init_spacing = get_init_spacings(freq)

        # Try our best to remove the dependence on init_spacings
#        init_diff = (init_spacing - init_spacing_0) * n_disk
        #init_rot = inv(eigendirections)[1:n_dim, :] * fill(init_diff, n_disk)
#        println("Init diff: $init_diff")

        _, spacing = optimize_bf_with_eigendirections(freq, optim_params, n_dim=n_dim,
#                                                      starting_point=vcat(fill(0., n_dim - 1),
#                                                                          init_diff),
                                                      eigendirections=eigendirections, n=64)


        push!(freqs, freq)
        push!(spacings, spacing)
    end
    s_0 = spacings[findall(f -> f == start_freq, freqs)[1]]
    freqs .-= start_freq
    spacings = map(x -> x - s_0, spacings)

    #fit_params = Vector{Vector{Float64}}()
    #for i in 1:n_dim
    #    fit = curve_fit(fit_function, freqs / 1.e9, map(x -> x[i] * 1e6, spacings),
    #                    #[i in [2, 3, 5] ? -1. : 1., 2., 10.])
    #                    [i in [1, 3, 5] ? -1. : 1., 2., 3.])
    #    push!(fit_params, coef(fit))
    #end
    write_json("$dir/shift_fit_$start_freq.json", Dict("freq" => start_freq, "n_dim" => n_dim,
                                                       "s_0" => s_0,
                                                       "freqs" => freqs,
                                                       "spacings" => spacings,
 #                                                      "fit_params" => fit_params,
                                                       "eigendirections" => eigendirections))
end

function ed_scan(freq_range; dir="results_ed_scan", cost_fun=nothing)
    eds = @distributed (vcat) for freq in freq_range
        calc_eigendirections(freq, cost_fun=cost_fun)
    end
    write_json("$dir/ed_scan.json", eds)
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
