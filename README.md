# Optimization tools for the BoostFractor.jl package

For gradient based optimization, the modified BoostFractor package at [https://github.com/David96/BoostFractor.jl](https://github.com/David96/BoostFractor.jl) is needed.

## Simple optimization

```julia

n_disk = 5
eps_disk = Complex(24)
# create regions from a vector of disk spacings
distance = distances_from_spacing([5e-3, 5e-3, 5e-3, 5e-3, 5e-3]; thickness=1e-3)
# Simulation range
freq_range = 15e9:1e6:20e9
# Range where the optimization runs on
# For β² optimization this is usually only a 50MHz range
# For reflectivity matching that's usually the full range
freq_optim = freq_range
# Epsilon starts with mirror, then air, then disk, then air, then disk..., then air
eps = vcat(1e20, reduce(vcat, [1, disk_eps] for i=1:n_disk), 1)

optimizer = init_optimizer(n_disk, 15e-2, # disk radius
                           1, 0, # Mmax & Lmax, only relevant for 3D
                           freq_range, freq_optim, distance, eps)

# Define which parameters we want to optimize
params = OrderedDict(:spacings => n_disk,
                     :air_loss => n_disk+1,
                     :disk_loss => n_disk,
                     :antenna => 1)

# Define our comparison function, that's called by the cost function,
# this can eg be the MSE to some reference (measurement)
ref = get_measurement_data()
function cmp(eout, p::BoosterParams)
   sum(abs2.(ref - eout[2, 1, :])) / length(ref)
end

# Run the optimization
res = optimize_spacings(optimizer;
                  cost_function=cost_fun(optimizer; parameters=params, cmp=cmp))
# Create new optimizer with the optimized parameters
p_new = apply_optim_res(optimizer, Optim.minimizer(res), params)
# Calculate boostfactor and reflection
eout = calc_eout(p_new, zeros(n_disk); reflect=true)
```
