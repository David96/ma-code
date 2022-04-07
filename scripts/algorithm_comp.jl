#!/usr/bin/env julia

using Distributed, SlurmClusterManager

if haskey(ENV, "SLURM_AVAILABLE") && ENV["SLURM_AVAILABLE"] == "true"
    addprocs(SlurmManager())
end

@everywhere begin
    using Thesis, OrderedCollections, LineSearches, Optim
end

n_disk = 20
n = 1
N = 1000
x_tol = 1e-6
g_tol = 1e-6
freq = 22e9

p0 = get_optim_params(freq, n_disk=n_disk, update_itp=false)
fit_params = OrderedDict(:spacings => n_disk)
cf = cost_fun(p0, 0, disable_constraints=false)

results = @distributed (vcat) for i = 1:N
    fixed_var = [2 .* (rand(n_disk).-0.5) .* 1e-4 for _=1:n]
    spacings_bfgs = optimize_spacings(p0, 0,
        algorithm=BFGS(linesearch=BackTracking(order=2)),
        options=Optim.Options(iterations=5000, f_tol=0, g_tol=0, x_tol=x_tol, show_trace=false),
        fixed_variation=fixed_var, cost_function=cf, n=n)
    spacings_nm = optimize_spacings(p0, 0,
        algorithm=NelderMead(),
        options=Optim.Options(iterations=5000, g_tol=g_tol, f_tol=0, x_tol=0, show_trace=false),
        fixed_variation=fixed_var, cost_function=cf, n=n)
    println("Finished optimization $i, bfgs vs nm: $(Optim.minimum(spacings_bfgs)) vs $(Optim.minimum(spacings_nm))")
    Dict(:bfgs => Dict(:spacings => Optim.minimizer(spacings_bfgs),
               :f_calls => Optim.f_calls(spacings_bfgs),
               :g_calls => Optim.g_calls(spacings_bfgs),
               :minimum => Optim.minimum(spacings_bfgs)),
     :nm => Dict(:spacings => Optim.minimizer(spacings_nm),
             :f_calls => Optim.f_calls(spacings_nm),
             :minimum => Optim.minimum(spacings_nm)))

end

write_json("scripts/results/bfgs_vs_nm.json",
           Dict(:freq => freq,
                :n => n,
                :N => N,
                :x_tol => x_tol,
                :g_tol => g_tol,
                :results => results))
