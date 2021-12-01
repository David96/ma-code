using BoostFractor
using Distances
using Interpolations
using DSP


"""
Calculates propagation matrices for all regions and frequencies at given relative spacings (optionally relative tilts aswell)
The returned matrix is (n_regions x n_freq x n_spacing x n_tilt_x x n_tilt_y x n_mode x n_mode)      
"""
function calc_propagation_matrices_grid(sbdry::SetupBoundaries,coords::CoordinateSystem, modes::Modes,spacing_grid,frequencies;tilt_x_grid=0,tilt_y_grid=0, prop=propagator, diskR=0.15)
    n_region = length(sbdry.distance)
    n_disk = (n_region-2)÷2
    n_freq = length(frequencies)
    n_spacing = length(spacing_grid)
    n_tilt_x = length(tilt_x_grid)
    n_tilt_y = length(tilt_y_grid)
    n_mode = modes.M*(2*modes.L+1)
    #Split booster into air gaps and solids(disks+mirror) since the latter stay constant and need just one calculation
    #for each frequency
    ind_gap = 2:2:n_region
    ind_solid = 1:2:n_region-1

    sbdry_gap = SeedSetupBoundaries(coords, diskno=n_disk, distance=sbdry.distance[ind_gap],
                                        epsilon=sbdry.eps[ind_gap],relative_tilt_x=sbdry.relative_tilt_x[ind_gap],
                                        relative_tilt_y=sbdry.relative_tilt_y[ind_gap],
                                        relative_surfaces=sbdry.relative_surfaces[ind_gap,:,:])

    sbdry_solid = SeedSetupBoundaries(coords, diskno=n_disk, distance=sbdry.distance[ind_solid],
                                        epsilon=sbdry.eps[ind_solid],relative_tilt_x=sbdry.relative_tilt_x[ind_solid],
                                        relative_tilt_y=sbdry.relative_tilt_y[ind_solid],
                                        relative_surfaces=sbdry.relative_surfaces[ind_solid,:,:])

    distance_0 = copy(sbdry_gap.distance)
    tilt_x_0 = copy(sbdry_gap.relative_tilt_x)
    tilt_y_0 = copy(sbdry_gap.relative_tilt_y)
    prop_matrix_grid = Array{Complex{Float64},7}(undef,n_region,n_freq,n_spacing,n_tilt_x,n_tilt_y,n_mode,n_mode)


    Threads.@threads for f in 1:n_freq
        prop_matrix_solid = calc_propagation_matrices(sbdry_solid,coords,modes;f=frequencies[f],prop=prop,diskR=diskR)
        for s in 1:n_spacing, tx in 1:n_tilt_x, ty in 1:n_tilt_y
            sbdry_i = copy_setup_boundaries(sbdry_gap,coords) #For thread safety
            sbdry_i.distance = distance_0 .+ spacing_grid[s]
            sbdry_i.relative_tilt_x = tilt_x_0 .+ tilt_x_grid[tx]
            sbdry_i.relative_tilt_y = tilt_y_0 .+ tilt_y_grid[ty]

            prop_matrix_gap = calc_propagation_matrices(sbdry_i,coords,modes;f=frequencies[f],prop=prop,diskR=diskR)
            prop_matrix_grid[ind_gap,f,s,tx,ty,:,:] = [(prop_matrix_gap[r][k,j]) for r in 1:n_disk+1, k in 1:n_mode, j in 1:n_mode]
            prop_matrix_grid[ind_solid,f,s,tx,ty,:,:] = [(prop_matrix_solid[r][k,j]) for r in 1:n_disk+1, k in 1:n_mode, j in 1:n_mode]
        end;
    end

    return prop_matrix_grid
end;


"""
Constructs the interpolation object from Interpolations without tilts
"""
function construct_prop_matrix_interpolation(prop_matrix_grid::Array{Complex{Float64},7}, spacing_grid)
    n_region = size(prop_matrix_grid,1)
    n_freq = size(prop_matrix_grid,2)
    n_spacing = size(prop_matrix_grid,3)
    n_tilt_x = size(prop_matrix_grid,4)
    n_tilt_y = size(prop_matrix_grid,5)
    n_mode = size(prop_matrix_grid,6)    
    #Construct the interpolation object
    itp_fun = Interpolations.BSpline(Cubic(Natural(OnCell())))
    itp = Interpolations.interpolate(prop_matrix_grid, (NoInterp(),NoInterp(),
                                        itp_fun,NoInterp(),
                                        NoInterp(),NoInterp(),NoInterp()))
    itp = Interpolations.scale(itp,1:n_region,1:n_freq,spacing_grid,1:n_tilt_x,1:n_tilt_y,1:n_mode,1:n_mode)  
    return itp
end;

"""
Calculate interpolated propagation matrices set without tilts
"""
function interpolate_prop_matrix(itp,dist_shift::AbstractArray{T,1}) where T<:Real
    n_region = size(itp,1)
    n_freq = size(itp,2)
    n_mode = size(itp,6)
    #Disk thickness stays constant and last air gap stay constant 
    dist_shift_all = [(r+1)%2==0 ? 0.0 : r==n_region ? 0 : dist_shift[r÷2] for r in 1:n_region]
    prop_matrix_set_interp = Array{Array{Complex{T},2}}(undef,n_region,n_freq)
    for f=1:n_freq
        prop_matrix_set_interp[:,f] = [itp(r,f,dist_shift_all[r],1,1,1:n_mode,1:n_mode) for r in 1:n_region]
    end
    return prop_matrix_set_interp
end;



"""
Calculates boostfactor for given frequencies and booster. Note that the distances in sbdry are meaningless since
propagation_matrices_set already contains the effect of spacings.
"""        
function calc_boostfactor_modes(sbdry,coords,modes, frequencies, prop_matrices_set::Array{Array{Complex{T},2},2}; diskR=0.15,prop=propagator) where T<:Real
    n_freq = length(frequencies)
    n_modes = size(prop_matrices_set[1,1])[1]
    EoutModes0 = Array{Complex{T},3}(undef,1,n_modes,n_freq)
    # Sweep over frequency
    Threads.@threads for f in 1:n_freq
        boost = transformer(sbdry,coords,modes; prop=prop,diskR=diskR,f=frequencies[f],propagation_matrices=prop_matrices_set[:,f],reflect=nothing)
        EoutModes0[1,:,f] = boost
    end
    return EoutModes0
end;

function calc_boostfactor_modes_jacobian(sbdry,coords,modes, frequencies, prop_matrices_set::Array{Array{Complex{T},2},2}; diskR=0.15,prop=propagator) where T<:Real
    n_freq = length(frequencies)
    n_modes = size(prop_matrices_set[1,1])[1]
    n_gaps = length(sbdry.eps)÷2
    EoutModes0 = Array{Complex{T},3}(undef,1,n_modes,n_freq)
    EoutModes0_jac = Array{Complex{T},3}(undef,n_gaps,n_modes,n_freq)

    # Sweep over frequency
    Threads.@threads for f in 1:n_freq
    #for f in 1:n_freq
        boost, boost_grad = transformer_gradient(sbdry,coords,modes; prop=prop,diskR=diskR,f=frequencies[f],propagation_matrices=prop_matrices_set[:,f],reflect=nothing)
        EoutModes0[1,:,f] =  boost
        EoutModes0_jac[:,:,f] = boost_grad
    end
    return EoutModes0, EoutModes0_jac
end;

"""
Calculates boostfactor and reflectivity
"""
function calc_modes(sbdry,coords,modes, frequencies, prop_matrices_set::Array{Array{Complex{T},2},2},reflect; diskR=0.15,prop=propagator) where T<:Real
    n_freq = length(frequencies)
    n_modes = size(prop_matrices_set[1,1])[1]
    EoutModes0 = Array{Complex{T},3}(undef,2,n_modes,n_freq)
    # Sweep over frequency
    Threads.@threads for f in 1:n_freq
        boost, refl = transformer(sbdry,coords,modes; prop=prop,diskR=diskR,f=frequencies[f],propagation_matrices=prop_matrices_set[:,f],reflect=reflect)
        EoutModes0[1,:,f] =  boost
        EoutModes0[2,:,f] =  refl
    end 
    return EoutModes0
end;


function calc_boostfactor_cost(dist_shift::AbstractArray{T, 1},itp,frequencies,sbdry::SetupBoundaries,
                coords::CoordinateSystem, modes::Modes, m_reflect; diskR=0.15, prop=propagator,
                ref=nothing, fix_phase=false,
                ref_comp=(eout, ref) -> sum(abs2.(eout[2, :, :] - ref))) where T<:Real
    dist_bound_hard = Interpolations.bounds(itp)[3]
    #Return hard penalty when exceeding interpolation bounds
    if any(.!(dist_bound_hard[1] .< dist_shift .< dist_bound_hard[2])) 
        return 1001.0
    end
    #Add soft penalty when approaching interpolation bounds
    penalty = soft_box_penalty(dist_shift,dist_bound_hard)

    prop_matrices_set_interp = interpolate_prop_matrix(itp,dist_shift);
    if ref === nothing
        Eout = calc_boostfactor_modes(sbdry,coords,modes,frequencies,prop_matrices_set_interp,diskR=diskR,prop=prop)
        cpld_pwr = abs2.(sum(conj.(Eout[1,:,:]).*m_reflect, dims=1)[1,:])
        cost =  -p_norm(cpld_pwr,-20)*penalty
    elseif ref !== nothing
        Eout = calc_modes(sbdry,coords,modes,frequencies,prop_matrices_set_interp, m_reflect,
                                      diskR=diskR,prop=prop)
        if fix_phase
            Eout[2, 1, :] .*= [exp(-1im * sum(dist_shift) / 3e8 * 2 * pi * f) for f in frequencies]
        end
        cost = ref_comp(Eout, ref)
    end
    return cost
end;

function calc_boostfactor_cost_gradient(dist_shift::AbstractArray{T, 1},itp,frequencies,sbdry::SetupBoundaries,coords::CoordinateSystem,modes::Modes,m_reflect; diskR=0.15, prop=propagator) where T<:Real
    dist_bound_hard = Interpolations.bounds(itp)[3]
    #Return hard penalty when exceeding interpolation bounds
    if any(.!(dist_bound_hard[1] .< dist_shift .< dist_bound_hard[2])) #|| !(dist_bound_hard[1]<-sum(dist_shift)<dist_bound_hard[2])
        return 1000.0, zeros(size(dist_shift))
    end
    #Add soft penalty when approaching interpolation bounds
    penalty = soft_box_penalty(dist_shift,dist_bound_hard)

    prop_matrices_set_interp = interpolate_prop_matrix(itp,dist_shift);
    Eout, Eout_jacobian = calc_boostfactor_modes_jacobian(sbdry,coords,modes,frequencies,prop_matrices_set_interp, diskR=diskR, prop=prop)
    cpld_amp = sum(conj.(Eout[1,:,:]).*m_reflect, dims=1)[1,:]
    cpld_pwr = abs2.(cpld_amp)
    cpld_amp_jacobian =  hcat([conj.(Eout_jacobian[:,:,f]) * m_reflect for f in 1:length(frequencies)]...)
    cpld_pwr_jacobian = real.(cpld_amp_jacobian.*transpose(conj.(cpld_amp)) + conj.(cpld_amp_jacobian).*transpose(cpld_amp))
    cost =  -p_norm(cpld_pwr,-20)*penalty
    cost_grad = -partial_p_norm(cpld_pwr,cpld_pwr_jacobian,-20)

    return cost, cost_grad
end;

"""
This adds a soft barrier penalty when dist_shift is approaching the maximum shifts allowed.
Hard box contrains usually confuse optimizers
""" 
function soft_box_penalty(shift::AbstractArray{T,1},bound_hard) where T<:Real
    soft_bound_depth = (bound_hard[2]-bound_hard[1])*0.05
    l_soft_bound = bound_hard[1] + soft_bound_depth
    u_soft_bound = bound_hard[2] - soft_bound_depth

    excess_pos = maximum([shift;sum(shift)]) - u_soft_bound
    excess_neg = -(minimum([shift;sum(shift)]) - l_soft_bound)
    excess = maximum([excess_pos,excess_neg,0])
    penalty = 1 - (excess/soft_bound_depth)^6
    return penalty
end





function surface_roughness(X,Y,ngaps; mag=1e-4,trunc=1e-2, xi=nothing, diskR=0.15)
    # TODO: Some kind of correlation length etc to have better control
    #over the power spectrum
    init = mag*[randn() for i in 1:ngaps, x in X, y in Y];
    # Trunctrate standard deviation such that huge outliers can't happen
    # TODO: This makes delta peaks at +- trunc with magnitude of what
    #we have left.numerical stable softmax
    # Better: redistribute this or directly use sampling from
    #trunctrated distribution
    if xi !== nothing
        # Now convolute with a gaussian
        g = [exp(-(x^2 + y^2)/(2*xi^2)) for x in X, y in Y]

        for i in 1:ngaps
        init[i,:,:] = conv2(init[i,:,:],
            g)[Int(floor(length(X)/2)):Int(floor(length(X)*1.5))-1,Int(floor(length(X)/2)):Int(floor(length(X)*1.5))-1]
        end

        # Now renormalize to have the rms mag
        init ./= sqrt.(sum(abs2.(init),
        dims=(2,3))./(length(X)*length(Y)))
        init .*= mag
    end

    init[init .> trunc] .= trunc
    init[init .< -trunc] .= -trunc

    init .*= [(x^2 + y^2) <= diskR^2 for i in 1:ngaps, x in X, y in Y]
    return init
end;

"""
Helper function to quickly copy SetupBoundaries
"""
function copy_setup_boundaries(sbdry::SetupBoundaries,coords::CoordinateSystem)
    n_disk = (length(sbdry.distance)-2)/2
    sbdry_new = SeedSetupBoundaries(coords, diskno=n_disk, distance=copy(sbdry.distance), epsilon=copy(sbdry.eps),relative_tilt_x=copy(sbdry.relative_tilt_x), relative_tilt_y=copy(sbdry.relative_tilt_y), relative_surfaces=copy(sbdry.relative_surfaces))
    return sbdry_new
end;

"""
p norm (https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions)
For large positive/negative p this is a differentiable approximatation of max(X)/min(X) 
Beware of numerical instability for large p. p~20 seems fine.
"""
function p_norm(X::AbstractArray{T,1},p) where T<:Real        
    magnitude = maximum(X)
    X_norm = X ./ magnitude
    return magnitude*(1/length(X) * sum(X_norm.^p))^(1/p)
end

function partial_p_norm(X,jacobian_X,p)
    magnitude = maximum(X)
    X_norm = X ./ magnitude
    jacobian_X_norm = jacobian_X 
    return length(X)^(-1/p).*(sum(X_norm.^p))^(1/p-1).*(sum(jacobian_X*(X_norm.^(p-1)),dims=2))
end

"""
Transformer Algorithm using Transfer Matrices and Modes to do the 3D Calculation.
"""
function transformer_gradient(bdry::SetupBoundaries, coords::CoordinateSystem, modes::Modes; f=10.0e9, velocity_x=0, prop=propagator, propagation_matrices::Array{Array{Complex{Float64},2},1}=Array{Complex{Float64},2}[], diskR=0.15, emit=BoostFractor.axion_induced_modes(coords,modes;B=nothing,velocity_x=velocity_x,diskR=diskR), reflect=nothing)
    # For the transformer the region of the mirror must contain a high dielectric constant,
    # as the mirror is not explicitly taken into account
    # To have same SetupBoundaries object for all codes and cheerleader assumes NaN, just define a high constant
    bdry.eps[isnan.(bdry.eps)] .= 1e30
    Nregions = length(bdry.eps)
    Ngaps = (Nregions)÷2

    #Definitions
    transmissionfunction_complete = [modes.id modes.zeromatrix ; modes.zeromatrix modes.id ]
    transmissionfunction_partials = [[modes.id modes.zeromatrix ; modes.zeromatrix modes.id ] for k in 1:Ngaps]
    lambda = wavelength(f)

    initial = emit


    axion_beam = Array{Complex{Float64}}(zeros((modes.M)*(2modes.L+1)))
    axion_beam_partials = [Array{Complex{Float64}}(zeros((modes.M)*(2modes.L+1))) for k in 1:Ngaps]
    
    k0 = 2pi/lambda*sqrt(bdry.eps[2])
    k_t = reshape(transpose(modes.mode_kt),modes.M*(2*modes.L+1))
    kz = diagm(sqrt.(k0^2 .- k_t.^2))
    diff_kz = -1im*[kz modes.zeromatrix ; modes.zeromatrix -kz]

    idx_reg(s) = Nregions-s+1


    for s in (Nregions-1):-1:1
        # Add up the summands of (M[2,1]+M[1,1]) E_0
        # (M is a sum over T_{s+1}^m S_s from s=1 to m) and we have just calculated
        #  T_{s+1}^m in the previous iteration)
        axion_beam .+= BoostFractor.axion_contrib(transmissionfunction_complete, sqrt(bdry.eps[idx_reg(s+1)]), sqrt(bdry.eps[idx_reg(s)]), initial, modes)

        # calculate T_s^m ---------------------------

        # Propagation matrix (later become the subblocks of P)
        diffprop = (isempty(propagation_matrices) ?
                        propagation_matrix(bdry.distance[idx_reg(s)], diskR, bdry.eps[idx_reg(s)], bdry.relative_tilt_x[idx_reg(s)], bdry.relative_tilt_y[idx_reg(s)], bdry.relative_surfaces[idx_reg(s),:,:], lambda, coords, modes; prop=prop) :
                        propagation_matrices[idx_reg(s)])



        # G_s P_s
        transmissionfunction_bdry = BoostFractor.get_boundary_matrix(sqrt(bdry.eps[idx_reg(s)]), sqrt(bdry.eps[idx_reg(s+1)]), diffprop, modes)     
        # T_s^m = T_{s+1}^m G_s P_s
        transmissionfunction_partials .*= [transmissionfunction_bdry]
        transmissionfunction_complete = transmissionfunction_partials[end]

        #insert ∂_k(G_k P_k) = G_k P_k diff_kz at correct position 
        transmissionfunction_partials[2*(1:Ngaps).==idx_reg(s)] .*= [diff_kz]   
     
        for k in 1:idx_reg(s)÷2#integer division is important
            #we only add those summands that contain ∂_k(G_k P_k) in their "T_s^m" 
            #since any ∂_k(T_s^m) not depending on P_k is zero
            axion_beam_partials[k] .+= BoostFractor.axion_contrib(transmissionfunction_partials[k], sqrt(bdry.eps[idx_reg(s+1)]), sqrt(bdry.eps[idx_reg(s)]), initial, modes)
        end
         
    end

    # The rest of 4.14a
    boost =  - (transmissionfunction_complete[BoostFractor.index(modes,2),BoostFractor.index(modes,2)]) \ (axion_beam)
    boost_grad =  Array{Complex{Float64}}(zeros(Ngaps,(modes.M)*(2modes.L+1)))
    for k in 1:Ngaps
        boost_grad[k,:] = -(transmissionfunction_complete[BoostFractor.index(modes,2),BoostFractor.index(modes,2)]) \ (-axion_beam_partials[k] .+ transmissionfunction_partials[k][BoostFractor.index(modes,2),BoostFractor.index(modes,2)]*boost)
    end
    
    # If no reflectivity is ought to be calculated, we only return the axion field
    if reflect === nothing
        return boost, boost_grad
    end

    refl = -transmissionfunction_complete[BoostFractor.index(modes,2),BoostFractor.index(modes,2)] \
           ((transmissionfunction_complete[BoostFractor.index(modes,2),BoostFractor.index(modes,1)]) * (reflect))
    refl_grad = Array{Complex{Float64}}(zeros(Ngaps,(modes.M)*(2modes.L+1)))
    for k in 1:Ngaps
        refl_grad[k,:] = -(transmissionfunction_complete[BoostFractor.index(modes,2),BoostFractor.index(modes,2)]) \ (transmissionfunction_partials[k][BoostFractor.index(modes,2),BoostFractor.index(modes,1)]*reflect .+ transmissionfunction_partials[k][BoostFractor.index(modes,2),BoostFractor.index(modes,2)]*refl)
    end
    return boost, boost_grad, refl, refl_grad
end

 ############################################################################

