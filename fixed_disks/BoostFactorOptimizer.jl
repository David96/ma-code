module BoostFactorOptimizer

using BoostFractor, LineSearches, ForwardDiff, Optim, Base.Threads

include("transformer_optim_utilities.jl") # Bunch of helper functions

export init_optimizer, optimize_spacings, calc_eout

mutable struct BoosterParams
    n_disk::Int
    epsilon::Float64
    diskR::Float64
    Mmax::Int
    Lmax::Int
    freq_center::Float64
    freq_width::Float64
    freq_optim
    freq_range
    coords
    sbdry_init
    modes
    m_reflect
    itp_sub
end

get_freq_optim(center, width) = range(center - width / 2, stop=center + width / 2, length=8)

function init_optimizer(n_disk, epsilon, diskR, Mmax, Lmax, freq_center, freq_width, freq_range,
                        distance, eps)
    freq_optim = get_freq_optim(freq_center, freq_width)
    coords = SeedCoordinateSystem(X = [1e-9], Y = [1e-9])
    n_regions = 2 * n_disk + 2
    thickness_var = zeros(n_regions, 1, 1)
    sbdry_init = SeedSetupBoundaries(coords, diskno=n_disk, distance=distance, epsilon=eps,
                                     relative_surfaces=thickness_var)
    modes = SeedModes(coords, ThreeDim=false, Mmax = Mmax, Lmax = Lmax, diskR = diskR)
    m_reflect = zeros(Mmax * (2 * Lmax + 1))
    m_reflect[Lmax + 1] = 1.0
    p = BoosterParams(n_disk, epsilon, diskR, Mmax, Lmax, freq_center, freq_width, freq_optim,
                      freq_range, coords, sbdry_init, modes, m_reflect, nothing)
    update_itp_sub(p)
    return p
end

function update_itp_sub(p::BoosterParams)
    spacing_grid = (-2500:50:2500)*1e-6
    prop_matrix_grid_sub = calc_propagation_matrices_grid(p.sbdry_init, p.coords, p.modes,
                                                          spacing_grid, p.freq_optim,
                                                          diskR=p.diskR, prop=propagator1D)
    p.itp_sub = construct_prop_matrix_interpolation(prop_matrix_grid_sub, spacing_grid)
end

function update_freq_center(params::BoosterParams, center::Float64)
    params.freq_optim = get_freq_optim(center, params.freq_width)
    update_itp_sub(params)
end

function cost_fun(p::BoosterParams, fixed_disk)
    return x -> begin
        # if one disk is fixed, insert it into dist_shift as the optimizer then runs on one
        # dimension less but we still need the "full" list of disks
        if fixed_disk > 0
            # it's important to *copy* and not modify x here otherwise the optimizer gets confused
            x = vcat(x[1:fixed_disk - 1], 0, x[fixed_disk:length(x)])
        end

        calc_boostfactor_cost(x, p.itp_sub, p.freq_optim, p.sbdry_init, p.coords,
                              p.modes, p.m_reflect, diskR=p.diskR, prop=propagator1D)
    end
end

function cost_fun_equidistant(p::BoosterParams)
    return x -> begin
        x_0 = fill(x[1], p.n_disk)
        calc_boostfactor_cost(x_0, p.itp_sub, p.freq_optim, p.sbdry_init, p.coords,
                              p.modes, p.m_reflect, diskR=p.diskR, prop=propagator1D)
    end
end

algorithm = BFGS(linesearch = BackTracking(order=3))
options = Optim.Options(f_tol = 1e-6)

function optimize_spacings(p::BoosterParams, fixed_disk::Int; starting_point=zeros(p.n_disk))
    spacings = Vector{Float64}()
    best_cost = Atomic{Float64}(1000.)
    cost_function = cost_fun(p, fixed_disk)
    lk = SpinLock()
    # Run initial optimization a few times and pick the best one
    @threads for i in 1:128
        # Add some random variation to start spacing.
        # Convergence very much depends on a good start point.
        x_0 = starting_point .+ 2 .* (rand(length(starting_point)).-0.5) .* 100e-6
        od = OnceDifferentiable(cost_function, x_0, autodiff=:forward)
        res = optimize(od, x_0, algorithm, options)
        cost = cost_function(Optim.minimizer(res))
        lock(lk)
        atomic_min!(best_cost, cost)
        if atomic_cas!(best_cost, cost, cost) === cost
            println("Best cost: $cost")
            spacings = Optim.minimizer(res)
        end
        unlock(lk)
    end
    spacings
end

function calc_eout(p::BoosterParams, spacings; fixed_disk=0)
    dist = copy(spacings)
    if fixed_disk > 0
        insert!(dist, fixed_disk, 0)
    end
    sbdry_optim = copy_setup_boundaries(p.sbdry_init, p.coords)
    sbdry_optim.distance[2:2:end-2] .+= dist

    #Calculate prop matrix grid at a dist shift of zero of optimized setup
    prop_matrix_grid_plot = calc_propagation_matrices_grid(sbdry_optim, p.coords, p.modes, 0,
                                                           p.freq_range, diskR=p.diskR,
                                                           prop=propagator1D)
    prop_matrix_plot = [prop_matrix_grid_plot[r,f,1,1,1,:,:] for r = 1:(2 * p.n_disk + 2),
                        f = 1:length(p.freq_range)]
    calc_boostfactor_modes(sbdry_optim, p.coords, p.modes, p.freq_range, prop_matrix_plot,
                           diskR=p.diskR, prop=propagator1D)
end

end
