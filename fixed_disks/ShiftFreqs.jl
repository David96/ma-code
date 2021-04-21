include("BoostFactorOptimizer.jl")
include("FileUtils.jl")

using .BoostFactorOptimizer
using BoostFractor
using PyPlot

# %%

freq_center = 21e9
freq_width = 50e6
freq_shift = 50e6

epsilon = 24
n_disk = 20
n_region = 2 * n_disk + 2

freq_0_spacing = read_optim_spacing_from_file("results/optim_2.1e10_2021-04-20T11:10.txt")

println("Loaded spacings from file: $freq_0_spacing")

freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)
eps = Array{Complex{Float64}}([i==1 ? 1e20 : i%2==0 ? 1 : epsilon for i=1:n_region])
distance = distances_from_spacing(freq_0_spacing)
optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, freq_width,
                              freq_range, distance, eps)
eout_0 = calc_eout(optim_params, zeros(n_disk))[1, 1, :]

plot(freq_range .* 1e-9, abs2.(eout_0))

# %%

for shift in 100e6:50e6:300e6
    update_freq_center(optim_params, freq_center + shift)

    spacings = Vector{Vector}(undef, n_disk)
    eout = Vector{Vector}(undef, n_disk)
    for i in 1:n_disk
        println("Optimizing for fixed disk $i at $(freq_center + shift)")
        spacings[i] = @time optimize_spacings(optim_params, i, starting_point=zeros(n_disk - 1))
        eout[i] = calc_eout(optim_params, spacings[i], fixed_disk=i)[1, 1, :]
    end

    for i in 1:n_disk
        write_optim_spacing_to_file(spacings[i], "$freq_center+$(shift)_f$i")
    end
end

# %%

clf()
legend_text = ["freq_0"]
plot(freq_range .* 1e-9, abs2.(eout_0))

for i in 1:n_disk
    push!(legend_text, "Fixed disk: $i")
    plot(freq_range .* 1e-9, abs2.(eout[i]))
end

xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")
legend(legend_text)
