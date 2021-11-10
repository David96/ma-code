using BoostFractor
using Glob
using Plots
using Statistics
using LsqFit
using Interact

plotly()
default(leg=false, titlefont=font(family="sans-serif", pointsize=10), markerstrokecolor=:auto)

n_disk = 20
n_region = 2 * n_disk + 2
epsilon = 24

eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)

function json_to_ed(json)
    reshape(reduce(vcat, convert(Vector{Vector{Float64}}, json)), (20, 20))
end

function plot_dims(dir, freq; area=true)
    #pyplot()
    ax1 = plot(ylabel="Boostfactor Magnitude", xlabel="Dimensions", xlims=(0.5, 20.5),
               xticks=1:20, leg=:bottomright)
    ax2 = plot(ylabel="Time [s]", xlabel="Dimensions", xlims=(0.5, 20.5), xticks=1:20,
               leg=:topleft)

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
        init_spacing = read_init_spacing_from_file("results/init_$(freq).txt")
        freq = data[1]["freq"]
        f_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9)
        distances = distances_from_spacing(init_spacing, n_region)
        p = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq, 50e6, f_range,
                                      distances, eps)
        for d in data
            spacings = convert(Vector{Float64}, d["spacing"])
            if haskey(d, "eigendirections") && d["n_dim"] > 0
                eigendirections = json_to_ed(d["eigendirections"])
                spacings = eigendirections[:, 1:d["n_dim"]] * spacings
            end
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
    ns = sort(collect(keys(x)), rev=true)
    for n in ns
        push!(l, "n=$n")
        p = sortperm(x[n])
        x_n = x[n][p]
        y_n = y[n][p]
        time_n = time[n][p]
        yerr_n = yerr[n][p]
        time_err_n = time_err[n][p]
        plot!(ax1, x_n[2:end], y_n[2:end], yerror=yerr_n[2:end], label="n=$n",
              markerstrokecolor=:auto)
        plot!(ax2, x_n[2:end], time_n[2:end], yerror=time_err_n[2:end], label="n=$n",
              markerstrokecolor=:auto)
        plot!(ax1, [1, 10, 20], [y_n[1], y_n[1], y_n[1]], yerror=[0, yerr_n[1], 0],
              label="Ref n=$n")
        plot!(ax2, [1, 10, 20], [time_n[1], time_n[1], time_n[1]], yerror=[0, time_err_n[1], 0],
              label="Ref n=$n")
    end
    map(display, [ax1, ax2])
end

function plot_freq_scan(start_freq, dir; area=true)
    x = Vector{Int}()
    x_ref = Vector{Int}()
    y = Vector{Float64}()
    y_ref = Vector{Float64}()
    for json in glob("spacings_*.json", dir)
        data = read_json(json)
        for d in data
            # The 20x20 Matrix is saved as a Vector{Vector{Any}} in JSON…
            # There might be easier ways of doing this.
            eigendirections = json_to_ed(d["eigendirections"])
            spacing = eigendirections[:, 1:d["n_dim"]] * d["spacing"]
            optim_params = get_optim_params(d["freq"])
            cost = -calc_real_bf_cost(optim_params, spacing, fixed_disk=0, area=area)
            if haskey(d, "ref") && d["ref"] == true
                push!(y_ref, cost)
                push!(x_ref, d["freq"])
            else
                push!(y, cost)
                push!(x, d["freq"])
            end
        end
    end
    plot(x, y, leg=true, ylabel="Boostfactor magnitude", label="Fixed eigendirections")
    plot!(x_ref, y_ref, label="Reference")
end

@. fit_function(x, p) = p[1] * (x - p[2])^2 + p[3]

function plot_eigendirections_fit(start_freq, dir="results_shift")
    data = read_json("$dir/shift_fit_$start_freq.json")
    spacings = convert(Vector{Vector{Float64}}, data["spacings"])
    #eigendirections = reshape(reduce(vcat, convert(Vector{Vector{Float64}},
    #                                               data["eigendirections"])), (20, 20))
    #s_0 = convert(Vector{Float64}, data["s_0"])
    freqs = -1.7e9:0.1e9:2e9
    spacings = spacings[length(spacings) - length(freqs) + 1:end]
    plot(leg=:outerright)
    for i in 1:data["n_dim"]
        plot!(freqs / 1e9, map(x -> x[i], spacings), label="Ed $i")
    end

    fit_params = Vector{Vector{Float64}}()
    for i in 1:data["n_dim"]
        fit = curve_fit(fit_function, freqs / 1.e9, map(x -> x[i] * 1e6, spacings),
                        [i in [4, 5] ? 3 : 1.5, 1., 3.])
        plot!(freqs / 1e9, fit_function(freqs / 1.e9, coef(fit)) / 1e6, label="Fit $i")
        push!(fit_params, coef(fit))
    end
    println(fit_params)
    return plot!()
end

function plot_fit_vs_opt(start_freq, fit_params, dir="results_shift"; area=true, plot_bf_at=-1)
    ref = read_json("$dir/spacings_ref_n=128_n-dim=5.json")
    fit = read_json("$dir/shift_fit_2.2e10.json")

    n_dim = ref[1]["n_dim"]
    ref_0 = ref[findall(r -> r["freq"] == start_freq, ref)[1]]
    s_0 = convert(Vector{Float64}, ref_0["spacing"])
    eigendirections_fit = json_to_ed(fit["eigendirections"])
    eigendirections_0 = json_to_ed(ref_0["eigendirections"])
    #p_0 = get_optim_params(start_freq)

    costs_ref = Vector{Float64}()
    costs_fit = Vector{Float64}()
    freqs = Vector{Float64}()
    for r in ref
        freq = r["freq"]
        eigendirections = json_to_ed(r["eigendirections"])
        spacing_ref = convert(Vector{Float64}, r["spacing"])
        p = get_optim_params(freq)
        cost_ref = -calc_real_bf_cost(p, eigendirections[:, 1:n_dim] * spacing_ref, area=area) / (area ? 50e6 : 1)

        shift_spacing = Vector{Float64}()
        for i in 1:length(fit_params)
            push!(shift_spacing, fit_function((freq - start_freq) / 1e9, fit_params[i]) * 1e-6)
        end
        spacing = eigendirections_fit[:, 1:length(fit_params)] * shift_spacing
        spacing += eigendirections_0[:, 1:length(s_0)] * s_0

        if freq == start_freq
            println("Shift spacing: $shift_spacing")
        end

        cost_fit = -calc_real_bf_cost(p, spacing, area=area) / (area ? 50e6 : 1)

        push!(costs_ref, cost_ref)
        push!(costs_fit, cost_fit)
        push!(freqs, freq)

        if plot_bf_at == freq
            eout_ref = calc_eout(p, eigendirections[:, 1:n_dim] * spacing_ref)
            eout_fit = calc_eout(p, spacing)
            p1 = plot(p.freq_range / 1e9, abs2.(eout_ref[1, 1, :]), label="Ref", leg=:topright)
            plot!(p1, p.freq_range / 1e9, abs2.(eout_fit[1, 1, :]), label="Fit")
            display(p1)
        end
    end
    p = sortperm(freqs)
    freqs = freqs[p]
    costs_ref = costs_ref[p]
    costs_fit = costs_fit[p]
    plot(freqs / 1e9, [costs_ref costs_fit], label=["Ref" "Fit"], leg=:outerright)
end

function plot_interactive_movement(start_freq, shift_range, s_0, eigendirections_0, 
        eigendirections_fit, fit_params)
    shifted_eout = Dict{Float64, Vector{Float64}}()
    freq_range = (start_freq + shift_range[1] * 1e9 - 0.5e9):0.004e9:(start_freq + shift_range[end] * 1e9 + 0.5e9)
    for shift in shift_range
        optim_params = get_optim_params(start_freq + shift * 1e9, freq_range=freq_range,
                                       update_itp=false)
        shift_spacing = Vector{Float64}()
        for i in 1:length(fit_params)
            push!(shift_spacing, fit_function(shift, fit_params[i]) * 1e-6)
        end
        spacing = eigendirections_fit[:, 1:length(fit_params)] * shift_spacing
        spacing += eigendirections_0[:, 1:length(s_0)] * s_0
        eout = calc_eout(optim_params, spacing, fixed_disk=0)
        shifted_eout[shift] = abs2.(eout[1, 1, :])
    end
    @manipulate for shift in shift_range
        vbox(plot(freq_range / 1e9, shifted_eout[shift], ylims=(0, 5e4)))
    end
end

function plot_diffs(dir)
    plot(leg=true)
    for json in glob("diffs_*.json", dir)
        println("Found file $json")
        data = read_json(json)
        x = data["M_range"]
        y = data["mean"]
        error = sqrt.(data["variance"])
        plot!(x, y, yerror=error, xlabel="M", ylabel="Difference",
              title="Eigenvalue differences at $(data["freq"])", label=data["variation"])
    end
    plot!()
end

function plot_eigendirections(freq, n; cost_fun=nothing,
        ev_ed = calc_eigendirections(freq, cost_fun=cost_fun, variation=50e-6, M=3000))
    ev = ev_ed[1]
    ed = ev_ed[2]
    #display(eigenvalues)
    #display(eigendirections)
    bar(collect(1:20), ev./ sum(ev),
        layout=(cld(n+1, 2), 2), subplot=1,
        size=(600, cld(n, 3) * 250), xlabel="\$i\$", ylabel="\$\\lambda_i \$", yscale=:log10,
        title="Eigenvalues")
    #eigendirections = gen_eigendirections()
    for i = 1:n
        bar!(collect(1:20), ed[:,i], subplot=i+1,
             xlabel="\$i\$", ylabel="\$d_i \$", ylims=(-0.6, 0.6), title="Eigendirection $i")
    end

    plot!()
end

function plot_ed_scan(file, freq_range)
    data = read_json(file)
    @manipulate for freq = freq_range
        i = indexin(freq, freq_range)[1]
        ev_ed = data[i]
        ev_ed[2] = json_to_ed(ev_ed[2])
        plot_eigendirections(freq, 5, ev_ed = ev_ed)
    end
end
