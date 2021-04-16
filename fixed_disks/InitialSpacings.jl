include("BoostFactorOptimizer.jl")

using .BoostFactorOptimizer
using PyPlot

freq_center = 20e9
freq_width= 50e6

epsilon = 24
n_disk = 20
n_region = 2 * n_disk + 2

c_0 = 2.99792458e8
#init_spacing = c_0 / freq_center / 2
#init_spacing = 0.007118 + (0.007118 - 0.008069)*(freq_center * 1e-9 - 22.288)
init_spacing = 0.008069

function distances_from_spacing(init_spacing)
    distance = Array{Float64}([i==1 ? 0 : i%2==0 ? init_spacing : 1e-3 for i=1:n_region])
    distance[end] = 0
    return distance
end

freq_range = (freq_center - 0.5e9):0.004e9:(freq_center + 0.5e9)
eps = Array{Complex{Float64}}([i==1 ? 1e20 : i%2==0 ? 1 : epsilon for i=1:n_region])
distance = distances_from_spacing(init_spacing)
optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, freq_width, freq_range,
                              distance, eps)

spacings = @time optimize_spacings(optim_params, 0)

#best_cost = 1000
#best = 0
#for i in 0.007:0.000001:0.009
#    global best, best_cost
#    cost = BoostFactorOptimizer.cost_fun_equidistant(optim_params)(i)
#    if cost < best_cost
#        println("Current best cost: $cost")
#        best_cost = cost
#        best = i
#    end
#end
#println("Best cost: $best_cost at $best")

eout = calc_eout(optim_params, spacings)

legend_text = ["freq_0"]
plot(freq_range .* 1e-9, abs2.(eout[1, 1, :]))
xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")
legend(legend_text)
