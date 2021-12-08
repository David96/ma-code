using BoostFractor, LineSearches, ForwardDiff, Optim, Base.Threads, DSP, Match

function distances_from_spacing(init_spacing::Float64, n_region::Int)
    distance = Array{Float64}([i==1 ? 0 : i%2==0 ? init_spacing : 1e-3 for i=1:n_region])
    distance[end] = 0
    return distance
end

function distances_from_spacing(init_spacing::Vector{Float64})
    vcat(0, reduce(vcat, [spacing, 1e-3] for spacing in init_spacing), 0)
end

# The distances are measured from disk_a-left to disk_b-left
struct BoosterConstraints
    min_dist_2_disks::Float64
    min_dist_8_disks::Float64
    max_dist_all_disks::Float64
end

mutable struct BoosterParams
    n_disk::Int
    epsilon::Complex
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
    prop
    constraints::BoosterConstraints
end

get_freq_optim(center, width; length=32) = range(center - width / 2, stop=center + width / 2,
                                                length=length)

function init_optimizer(n_disk, epsilon, diskR, Mmax, Lmax, freq_center, freq_width, freq_range,
        distance, eps; three_dims=false, constraints=BoosterConstraints(6e-3, 48e-3, 304e-3),
        update_itp=true)
    freq_optim = get_freq_optim(freq_center, freq_width)
    if three_dims
        dx = 0.007
        X = -0.5:dx:0.5
        Y = -0.5:dx:0.5
        coords = SeedCoordinateSystem(X = X, Y = Y)
    else
        coords = SeedCoordinateSystem(X = [1e-9], Y = [1e-9])
    end
    n_regions = 2 * n_disk + 2
    thickness_var = zeros(n_regions, 1, 1)
    sbdry_init = SeedSetupBoundaries(coords, diskno=n_disk, distance=distance, epsilon=eps,
                                     relative_surfaces=thickness_var)
    modes = SeedModes(coords, ThreeDim=three_dims, Mmax = Mmax, Lmax = Lmax, diskR = diskR)
    m_reflect = zeros(Mmax * (2 * Lmax + 1))
    m_reflect[Lmax + 1] = 1.0
    p = BoosterParams(n_disk, epsilon, diskR, Mmax, Lmax, freq_center, freq_width, freq_optim,
                      freq_range, coords, sbdry_init, modes, m_reflect, nothing,
                      three_dims ? propagator : propagator1D, constraints)
    if update_itp
        update_itp_sub(p)
    end
    return p
end

function update_itp_sub(p::BoosterParams; range=-2500:50:2500, losses=0:0)
    spacing_grid = (range)*1e-6
    prop_matrix_grid_sub = calc_propagation_matrices_grid(p.sbdry_init, p.coords, p.modes,
                                                          spacing_grid, p.freq_optim,
                                                          losses_grid=losses,
                                                          diskR=p.diskR, prop=p.prop)
    p.itp_sub = construct_prop_matrix_interpolation(prop_matrix_grid_sub, spacing_grid,
                                                   losses)
end

function update_freq_center(params::BoosterParams, center = params.freq_center,
        width = params.freq_width; update_itp=true, length=32)
    params.freq_optim = get_freq_optim(center, width, length=length)
    if update_itp
        update_itp_sub(params)
    end
end

function update_distances(params::BoosterParams, distances::Vector; update_itp=true)
    params.sbdry_init.distance = distances
    if update_itp
        update_itp_sub(params)
    end
end

function get_distance(a::Int, b::Int, p::BoosterParams, spacings)
    return a < b ? (sum(p.sbdry_init.distance[2a + 1:2b]) + sum(spacings[a + 1:b])) :
                   (sum(p.sbdry_init.distance[2b + 1:2a]) + sum(spacings[b + 1:a]))
end

function get_penalty(disk1, disk2, p, x, dist; is_min_dist=true)
    dist_rel = is_min_dist ? (get_distance(disk1, disk2, p, x) - dist) :
                             (dist - get_distance(disk1, disk2, p, x))
    #println("Distance between: $disk1 and $disk2: $(get_distance(disk1, disk2, p, x))")
    if dist_rel < 0
        return Inf
    elseif dist_rel < 1e-3
        return (dist_rel * 1e3) ^ 6
    end
    return 0
end

function apply_constraints(p::BoosterParams, x; debug=false)
    penalty = 0.
    # Disks are additionally constraint because every 8th disk is on the same rail.
    # Also, two disks have a minimum distance and the booster has a fixed length
    for d = 1:(p.n_disk - 1)
        p1 = d <= p.n_disk - 8 ?
                get_penalty(d, d + 8, p, x, p.constraints.min_dist_8_disks) : 0
        p2 = get_penalty(d, d + 1, p, x, p.constraints.min_dist_2_disks)
        if debug
            println("Applying penalty of $(p1 + p2)")
        end
        if p1 == Inf || p2 == Inf
            if debug
                println("$(p1 == Inf ? "8th" : "2nd") disk constraint")
            end
            return Inf
        else
            penalty += p1 + p2
        end
    end
    p_length = get_penalty(1, p.n_disk, p, x, p.constraints.max_dist_all_disks,
                           is_min_dist=false)
    if p_length == Inf
        if debug
            println("Booster length constraint")
        end
        return Inf
    else
        penalty += p_length
    end
    return penalty
end

function cost_fun(p::BoosterParams, fixed_disk; gradient=false, disable_constraints=false,
                  kwargs...)
    return x -> begin
        penalty = 0.
        # if one disk is fixed, insert it into dist_shift as the optimizer then runs on one
        # dimension less but we still need the "full" list of disks
        if fixed_disk > 0
            rel_pos = -sum(x[1:fixed_disk - 1])
            # make sure disks don't move past each other
            if -rel_pos >= p.sbdry_init.distance[2 * fixed_disk]
                #println("Damn: $rel_pos vs $(p.sbdry_init.distance)")
                return (-rel_pos) - p.sbdry_init.distance[2 * fixed_disk]
            end
            # it's important to *copy* and not modify x here otherwise the optimizer gets confused
            x = vcat(x[1:fixed_disk - 1], rel_pos, x[fixed_disk:length(x)])
        end

        penalty = disable_constraints ? 0. : apply_constraints(p, x)
        if penalty == Inf
            return 1000.
        end

        if gradient
            cost, grad = calc_boostfactor_cost_gradient(x, p.itp_sub, p.freq_optim, p.sbdry_init,
                                                        p.coords, p.modes, p.m_reflect,
                                                        diskR=p.diskR, prop=p.prop)
            cost + penalty, grad
        else
            calc_boostfactor_cost(x, p.itp_sub, p.freq_optim, p.sbdry_init, p.coords, p.modes,
                                  p.m_reflect; diskR=p.diskR, prop=p.prop, kwargs...) + penalty
        end
    end
end

function cost_fun_rot(p::BoosterParams, eigendirections; kwargs...)
    cf = cost_fun(p, 0; kwargs...)
    return x -> begin
        x_r = eigendirections[:,1:length(x)] * x
        return cf(x_r)
    end
end

function cost_fun_equidistant(p::BoosterParams)
    return x -> begin
        x_0 = fill(x[1], p.n_disk)
        calc_boostfactor_cost(x_0, p.itp_sub, p.freq_optim, p.sbdry_init, p.coords,
                              p.modes, p.m_reflect, diskR=p.diskR, prop=p.prop)
    end
end

function optimize_spacings(p::BoosterParams, fixed_disk::Int; starting_point=zeros(p.n_disk),
                           cost_function=cost_fun(p, fixed_disk), n=1024,
                           algorithm=BFGS(linesearch=BackTracking(order=2)),
                           options=Optim.Options(f_tol=1e-6),
                           threshold_cost=-Inf)
    spacings = Vector{Float64}()
    best_cost = Atomic{Float64}(1002.)
    stop = Atomic{Bool}(false)
    lk = SpinLock()
    # Run initial optimization a few times and pick the best one
    @threads for i in 1:n
        if !stop[]
            # Add some random variation to start spacing.
            # Convergence very much depends on a good start point.
            x_0 = starting_point .+ 2 .* (rand(length(starting_point)).-0.5) .* 100e-6

            # Depending on the optimizer we want a differentiable cost function
            cf = @match algorithm begin
                    _::Optim.ZerothOrderOptimizer => cost_function
                    _::Optim.FirstOrderOptimizer => OnceDifferentiable(cost_function, x_0,
                                                                       autodiff=:forward)
                    _::Optim.SecondOrderOptimizer => TwiceDifferentiable(cost_function, x_0,
                                                                       autodiff=:forward)
            end
            res = optimize(cf, x_0, algorithm, options)
            cost = cost_function(Optim.minimizer(res))
            lock(lk)
            atomic_min!(best_cost, cost)
            if atomic_cas!(best_cost, cost, cost) === cost
                spacings = Optim.minimizer(res)
                if cost < threshold_cost
                    println("Reached threshold at $i")
                    stop[] = true
                end
            end
            unlock(lk)
        end
    end
    println("Best cost: $best_cost")
    spacings
end

spacings_with_fd(spacings, fd) = vcat(spacings[1:fd - 1],
                                      -sum(spacings[1:fd - 1]),
                                      spacings[fd:end])

function calc_real_bf_cost(p::BoosterParams, spacings; fixed_disk=0, area=true)
    if fixed_disk > 0
        spacings = spacings_with_fd(spacings, fixed_disk)
    end
    penalty = apply_constraints(p, spacings)
    if penalty == Inf
        return 1000
    end
    p1 = deepcopy(p)
    p1.freq_range = p1.freq_optim
    eout = calc_eout(p1, spacings, fixed_disk=0)
    pwr = abs2.(sum(conj.(eout[1,:,:]) .* p.m_reflect, dims=1)[1,:])
    if area
        -sum(pwr) * p1.freq_range.step
    else
        -minimum(pwr)
    end
end

function calc_eout(p::BoosterParams, spacings; fixed_disk=0, reflect=false,
        disable_constraints=false)
    if fixed_disk > 0
        spacings = spacings_with_fd(spacings, fixed_disk)
    end
    sbdry_optim = copy_setup_boundaries(p.sbdry_init, p.coords)
    sbdry_optim.distance[2:2:end-2] .+= spacings

    # check that disks didn't move past each other
    if count(x -> x < 0, sbdry_optim.distance[end - 1]) > 0
        println("We fucked, that's not possible!")
        throw(ArgumentError("Negative relative spacings aren't possible!"))
    end

    if !disable_constraints && apply_constraints(p, spacings) == Inf
        throw(ArgumentError("Spacings don't fullfill constraints!"))
    end

    #Calculate prop matrix grid at a dist shift of zero of optimized setup
    prop_matrix_grid_plot = calc_propagation_matrices_grid(sbdry_optim, p.coords, p.modes, 0,
                                                           p.freq_range, diskR=p.diskR,
                                                           prop=p.prop)
    prop_matrix_plot = [prop_matrix_grid_plot[r,f,1,1,1,:,:] for r = 1:(2 * p.n_disk + 2),
                        f = 1:length(p.freq_range)]
    if reflect
        calc_modes(sbdry_optim, p.coords, p.modes, p.freq_range, prop_matrix_plot, p.m_reflect,
                   diskR=p.diskR, prop=p.prop)
    else
        calc_boostfactor_modes(sbdry_optim, p.coords, p.modes, p.freq_range, prop_matrix_plot,
                               diskR=p.diskR, prop=p.prop)
    end
end


function get_phase_depth(f; eps=24, d=1e-3)
    2π * f * d * sqrt(eps) / 3e8
end

function get_freq(phase_depth; eps=24, d=1e-3)
    phase_depth * 3e8 / (2π * d * sqrt(eps))
end

function get_init_spacings(freq, freq_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9),
                           epsilon=24, n_disk=20)
    if epsilon == 24 && n_disk == 20
        p = [0.04323293823102593, 0.11927287669916398, 0.004077191528864808]
        return p[1] * exp(-p[2] * freq / 1e9) + p[3]
    end
    if isfile("results/init_$(freq).txt")
        return read_init_spacing_from_file("results/init_$(freq).txt")
    else
        eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
        optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq, 50e6, freq_range,
                                      distances_from_spacing(0.0, 2*n_disk + 2), eps,
                                      update_itp=false)
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
        write_init_spacing_to_file(best_spacing, freq)
        return best_spacing
    end
end

function get_optim_params(freq; freq_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9),
                          update_itp=true, epsilon=24, n_disk=20, freq_width=50e6)

    eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
    init_spacing = get_init_spacings(freq)
    distances = distances_from_spacing(init_spacing, n_disk * 2 + 2)
    optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq, freq_width, freq_range,
                                  distances, eps, update_itp=update_itp,
                                  constraints = BoosterConstraints(0., 0., 100.))

    return optim_params
end

function group_delay(refl, df)
    -diff(unwrap(angle.(refl))) ./ (2*pi*df)
end
