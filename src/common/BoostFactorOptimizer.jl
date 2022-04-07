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
    diskR::Float64
    Mmax::Int
    Lmax::Int
    freq_center::Float64
    freq_width::Float64
    freq_optim
    freq_range
    coords
    sbdry_init
    σ_mirror
    modes
    m_reflect
    itp_sub
    prop
    constraints::BoosterConstraints
end

get_freq_optim(center, width; length=32) = range(center - width / 2, stop=center + width / 2,
                                                length=length)

function init_optimizer(n_disk, diskR, Mmax, Lmax, freq_center, freq_width, freq_range,
        distance, eps, σ_mirror=1e30; three_dims=false, constraints=BoosterConstraints(6e-3, 48e-3, 304e-3),
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
    thickness_var = zeros(ComplexF64, n_regions, 1, 1)
    sbdry_init = SeedSetupBoundaries(coords, diskno=n_disk, distance=distance, epsilon=eps,
                                     relative_surfaces=thickness_var)
    modes = SeedModes(coords, ThreeDim=three_dims, Mmax = Mmax, Lmax = Lmax, diskR = diskR)
    m_reflect = zeros(Mmax * (2 * Lmax + 1))
    m_reflect[Lmax + 1] = 1.0
    p = BoosterParams(n_disk, diskR, Mmax, Lmax, freq_center, freq_width, freq_optim,
                      freq_range, coords, sbdry_init, σ_mirror, modes, m_reflect, nothing,
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
        return (-dist_rel * 1e6) ^ 2
    #elseif dist_rel < 1e-3
    #    return ((1e-3 - dist_rel) * 1e3) ^ 6
    end
    return 0
end

function apply_constraints(p::BoosterParams, x; debug=false)
    penalty = 0.
    d1 = p.sbdry_init.distance[2] + x[1]
    if d1 < 0
        penalty += (-d1 * 1e6) ^ 2
        if debug
            println("Disk <-> Mirror $(ForwardDiff.value(penalty))")
        end
    end
    # Disks are additionally constraint because every 8th disk is on the same rail.
    # Also, two disks have a minimum distance and the booster has a fixed length
    for d = 1:(p.n_disk - 1)
        p1 = d <= p.n_disk - 8 ?
                get_penalty(d, d + 8, p, x, p.constraints.min_dist_8_disks) : 0
        p2 = get_penalty(d, d + 1, p, x, p.constraints.min_dist_2_disks)
        if debug && p1 + p2 > 0
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
    p_length = get_penalty(0, p.n_disk, p, x, p.constraints.max_dist_all_disks,
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

function min_cost(eout, p::BoosterParams)
    cpld_pwr = abs2.(sum(conj.(eout[1,:,:]).*p.m_reflect, dims=1)[1,:])
    -p_norm(cpld_pwr,-20)
end

function calc_boostfactor_cost(dist_shift::AbstractArray{T, 1}, p::BoosterParams;
                cmp=min_cost,
                parameters::OrderedDict=OrderedDict(:spacings => length(dist_shift)),
                scaling=fill(1, length(dist_shift))) where T<:Real
    #dist_bound_hard = Interpolations.bounds(itp)[3]
    dist_shift_scaled = dist_shift .* scaling
    params = extract_params(dist_shift_scaled, parameters)

    if false
        println("Shift: $(ForwardDiff.value.(dist_shift))")
        #println("Spacings: $(ForwardDiff.value.(params[:spacings]))")
        println("Loss: $(ForwardDiff.value.(params[:loss]))")
        println("σ_mirror: $(ForwardDiff.value.(params[:σ_mirror]))")
        println("Antenna: $(ForwardDiff.value.(params[:antenna]))")
        println("Scalings: $scaling")
    end

    penalty = 0

    # Make sure to create SetupBoundaries{Real} to fit the ForwardDiff.Dual type
    eps_new = Array{Complex{Real}, 1}(p.sbdry_init.eps)
    dist_new = Array{Real, 1}(p.sbdry_init.distance)
    p_new = deepcopy(p)
    p_new.freq_range = p_new.freq_optim
    p_new.sbdry_init = SeedSetupBoundaries(p.coords, diskno=p.n_disk, distance=dist_new,
                                           epsilon=eps_new)
    apply_optim_res!(p_new, params)
    if :loss in keys(params)
        for eps in p_new.sbdry_init.eps[2:end]
            loss = imag(sqrt(eps))
            if loss > 0
                penalty += (loss * 1e3)^6
            end
        end
    end
    if :antenna in keys(params)
        if params[:antenna][1] > 0.5
            penalty += (params[:antenna][1] - 0.5)^2
        end
    end
    #if :spacings in keys(params)
    #    for dst in p_new.sbdry_init.distance[1:end-1]
    #        if dst < 0
    #            penalty += (dst * 1e4)^4
    #            println("Negative spacing spotted, applying $penalty")
    #        end
    #    end
    #end
    # we don't want constraints here as they're already taken into account for our penalty and
    # calc_eout just stops if they are disregarded
    eout = calc_eout(p_new, zeros(p_new.n_disk), reflect=true, disable_constraints=true)
    cost = cmp(eout, p) + penalty
    if cost == NaN
        println("Got invalid res, penalty: $penalty, input: $(ForwardDiff.value.(dist_shift_scaled))")
        return Inf
    end
    return cost
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
            return Inf#1000.
        end

        if gradient
            cost, grad = calc_boostfactor_cost_gradient(x, p)
            cost + penalty, grad
        else
            calc_boostfactor_cost(x, p; kwargs...) + penalty
        end
    end
end

function cost_fun_rot(eigendirections, cf)
    return x -> begin
        x_r = eigendirections[:,1:length(x)] * x
        return cf(x_r)
    end
end

function optimize_spacings(p::BoosterParams, fixed_disk::Int;
                           starting_point=zeros(p.n_disk),
                           variation=100e-6,
                           cost_function=cost_fun(p, fixed_disk), n=1024,
                           algorithm=BFGS(linesearch=BackTracking(order=2)),
                           options=Optim.Options(f_tol=1e-6),
                           threshold_cost=-Inf,
                           fixed_variation=nothing)
    best_cost = Atomic{Float64}(Inf)
    best_res = nothing
    stop = Atomic{Bool}(false)
    lk = ReentrantLock()
    # Run initial optimization a few times and pick the best one
    @threads for i in 1:n
        if !stop[]
            # Add some random variation to start spacing.
            # Convergence very much depends on a good start point.
            if fixed_variation !== nothing
                x_0 = starting_point .+ fixed_variation[i]
            else
                x_0 = starting_point .+ 2 .* (rand(length(starting_point)).-0.5) .* variation
            end

            # Depending on the optimizer we want a differentiable cost function
            cf = @match algorithm begin
                    _::Optim.ZerothOrderOptimizer => cost_function
                    _::Optim.FirstOrderOptimizer => OnceDifferentiable(cost_function, x_0,
                                                                       autodiff=:forward)
                    _::Optim.SecondOrderOptimizer => TwiceDifferentiable(cost_function, x_0,
                                                                       autodiff=:forward)
            end
            res = optimize(cf, x_0, algorithm, options)
            #display(res)
            cost = cost_function(Optim.minimizer(res))
            atomic_min!(best_cost, cost)
            if atomic_cas!(best_cost, cost, cost) === cost
                lock(lk) do
                    best_res = res
                    if cost < threshold_cost
                        println("Reached threshold at $i")
                        stop[] = true
                    end
                end
            end
        end
    end
    display(best_res)
    println("Best cost: $best_cost")
    best_res
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
    if !disable_constraints && count(x -> x < 0, sbdry_optim.distance[1:end - 1]) > 0
        println("We fucked, that's not possible!")
        throw(ArgumentError("Negative relative spacings aren't possible!"))
    end

    if !disable_constraints && apply_constraints(p, spacings) == Inf
        throw(ArgumentError("Spacings don't fullfill constraints!"))
    end

    #Calculate prop matrix grid at a dist shift of zero of optimized setup
    n_region = length(sbdry_optim.eps)
    n_freq = length(p.freq_range)
    prop_matrices = Array{Array{Complex{Real}, 2}, 2}(undef, n_region, n_freq)
    for (i, f) in enumerate(p.freq_range)
        sbdry_optim.eps[1] = complex(1, p.σ_mirror / (2pi * f))
        prop_matrices[:, i] = calc_propagation_matrices(sbdry_optim, p.coords, p.modes;
                                                        f=f, prop=p.prop, diskR=p.diskR)
    end
    if reflect
        calc_modes(sbdry_optim, p.coords, p.modes, p.freq_range, prop_matrices, p.m_reflect,
                   diskR=p.diskR, prop=p.prop)
    else
        calc_boostfactor_modes(sbdry_optim, p.coords, p.modes, p.freq_range, prop_matrices,
                               diskR=p.diskR, prop=p.prop)
    end
end


function get_phase_depth(f; eps=24, d=1e-3)
    2π * f * d * sqrt(eps) / 3e8
end

function get_freq(phase_depth; eps=24, d=1e-3)
    phase_depth * 3e8 / (2π * d * sqrt(eps))
end

function get_init_spacings(freq, freq_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9);
                           epsilon=24, n_disk=20)
    if epsilon == 24 && n_disk == 20 && false
        p = [0.04323293823102593, 0.11927287669916398, 0.004077191528864808]
        return p[1] * exp(-p[2] * freq / 1e9) + p[3]
    #end
    #if isfile("results/init_$(freq).txt")
    #    return read_init_spacing_from_file("results/init_$(freq).txt")
    else
        eps = vcat(1e20, n_disk == 0 ? [] : reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
        n_region = length(eps)
        optim_params = init_optimizer(n_disk, 0.15, 1, 0, freq, 50e6, freq_range,
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
        #write_init_spacing_to_file(best_spacing, freq)
        return best_spacing
    end
end

function get_optim_params(freq; freq_range = (freq - 0.5e9):0.004e9:(freq + 0.5e9),
                          update_itp=true, epsilon=complex(24, 0), n_disk=20, freq_width=50e6)

    eps = vcat(1e20, n_disk == 0 ? [] : reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
    init_spacing = get_init_spacings(freq, n_disk=n_disk, epsilon=epsilon)
    distances = distances_from_spacing(init_spacing, n_disk * 2 + 2)
    optim_params = init_optimizer(n_disk, 0.15, 1, 0, freq, freq_width, freq_range,
                                  distances, eps, update_itp=update_itp)

    return optim_params
end

function apply_optim_res(p::BoosterParams, args...)
    tmp = deepcopy(p)
    apply_optim_res!(tmp, args...)
    tmp
end

apply_optim_res!(p::BoosterParams, res, fit_params) = apply_optim_res!(p,
                                                            extract_params(res, fit_params))
function apply_optim_res!(p::BoosterParams, params)
    if :spacings in keys(params)
        p.sbdry_init.distance[2:2:end-1] .+= params[:spacings]
    end
    if :loss in keys(params)
        losses = params[:loss]
        for (i, r) in enumerate(2:1:(length(p.sbdry_init.eps)))
            p.sbdry_init.eps[r] = (sqrt(p.sbdry_init.eps[r]) + complex(0, losses[i]))^2
        end
    end
    if :ren in keys(params)
        rens = params[:rens]
        for (i, r) in enumerate(2:(length(p.sbdry_init.eps)))
            p.sbdry_init.eps[r] = (sqrt(p.sbdry_init.eps[r]) + complex(rens[i], 0))^2
        end
    end
    if :antenna in keys(params)
        p.sbdry_init.distance[end] += params[:antenna][1]
    end
    if :σ_mirror in keys(params)
        p.σ_mirror = params[:σ_mirror][1]
    end

    # The antenna position is special. Its *absolute* position is important for the
    # overall phase therefore if we change the spacings we subtract the change here
    if :spacings in keys(params)
        p.sbdry_init.distance[end] -= sum(params[:spacings])
    end
end

function extract_params(res, fit_params)
    params = Dict()
    offset = 1
    for (param, n) in fit_params
        params[param] = res[offset:offset+n-1]
        offset += n
    end
    params
end

function group_delay(refl, df)
    -diff(unwrap(angle.(refl))) ./ (2*pi*df)
end
