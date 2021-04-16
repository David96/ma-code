# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Julia 1.6.0
#     language: julia
#     name: julia-1.6
# ---

# %% [markdown]
# # Introduction
#
# In this notebook we will optimize disk spacings using the transformer algorithm. When optimizing for a certain frequency range $\nu_{min} - \nu_{max}$
# we minimize the cost function $C=-min(\beta ^2 (\nu) | \nu_{min} < \nu < \nu_{max})$. Essentially we take the worst frequency point of the boostfactor curve in this range and try to increase it (by decreasing the negative). This will naturally lead to a flat top hat shape.
#
# To reduce computation we only calculate the boostfactor in the range $\nu_{min} - \nu_{max}$ and keep the number of frequency points to a sensible minimum. With increasing disk number more points are needed to resolve the "wiggles" on top of the boostfactor. 
#
# One also has to choose what boostfactor mode to optimize for. Fundamental mode, other modes, coupled power?
#
# Transformer calculates the mixing between modes which is computationally the most expensive part. In order to make optimization feasable we calculate the mixing for a selected number of points beforehand and then quickly interpolate the exact spacing during optimization. 
#
# For gradient based methods we have to approximate the non-differentiable minimum() with a smooth function like the p-norm.
# Still, very close to the minimum the gradient might be unstable and gradient based methods might struggle. 
#

# %%
#push!(LOAD_PATH, pwd() * "/../src")
using BoostFractor
using LineSearches, Optim
using ForwardDiff #If we use gradient-based optimization
include("transformer_optim_utilities.jl") #Bunch of helper functions
using PyPlot
using Base.Threads
using Dates
using DelimitedFiles

# %% [markdown]
# # Setup

# %%
#Physical Parameters
n_disk = 20
epsilon = 24
diskR = 0.15
Mmax = 1
Lmax = 0
#We start with the most generic spacing possible: Equidistant spacing. Make sure the peak is roughly located in the desired 
#frequency range.
init_spacing = 0.007118

# %%
#Frequency range to optimize. For 20 disks 8 points seems sufficient, 80 disks needs ~20-25
freq_center = 22.288e9
#freq_center = 20.e9
freq_width = 50e6
n_freq_optim = 8
#freq_min = freq_center-freq_width/2
#freq_max = freq_center+freq_width/2
frequencies_optim(center, width) = range(center - width / 2,stop=center + width / 2,
                                         length=n_freq_optim);
c_0 = 2.99792458e8
λ1 = c_0 / (freq_center * sqrt(epsilon))
λ2 = c_0 / freq_center
#println("λ1: $λ1 λ2: $λ2, $(λ1/sqrt(epsilon))")
#init_spacing = 0.00808824
#println(init_spacing)

# %%
#For Plotting
df = 0.004*1e9
frequencies_plot = 22.1e9:df:22.6e9;
#frequencies_plot = 19.0e9:df:22.5e9

#Usual transformer setup
# 1D Coordinate System
X = [1e-9]
Y = [1e-9]
coords = SeedCoordinateSystem(X = X, Y = Y)
n_region = 2*n_disk+2
prop = propagator1D

# SetupBoundaries (note that this expects the mirror to be defined explicitly as a region)
eps = Array{Complex{Float64}}([i==1 ? 1e20 : i%2==0 ? 1 : epsilon for i=1:n_region])

distance = Array{Float64}([i==1 ? 0 : i%2==0 ? init_spacing : 1e-3 for i=1:n_region])
distance[end] = 0

#Uncomment to include thickness varation
init_thickness_variation = zeros(n_region,length(X),length(Y))
#xi = 0.035
#sigma = 2e-6;
#init_thickness_variation = surface_roughness(X,Y,n_region,mag=sigma,trunc=sigma*10, xi=xi, diskR=diskR)

sbdry_init = SeedSetupBoundaries(coords, diskno=n_disk, distance=distance, epsilon=eps,
                                 relative_surfaces=init_thickness_variation)

# Initialize modes
modes = SeedModes(coords, ThreeDim=false, Mmax=Mmax, Lmax=Lmax, diskR=diskR)

#  Mode-Vector defining beam shape to be reflected on the system
m_reflect = zeros(Mmax*(2*Lmax+1))
m_reflect[Lmax+1] = 1.0;

# # Plot Initial Boostfactor 

#Calculate FULL frequency prop matrix of initial setup for plotting
prop_matrix_grid_plot = calc_propagation_matrices_grid(sbdry_init, coords, modes, 0,
                                                       frequencies_plot; diskR=diskR,
                                                       prop=prop);
prop_matrix_plot = [prop_matrix_grid_plot[r,f,1,1,1,:,:] for r=1:n_region, f=1:length(frequencies_plot)]
Eout_init = calc_boostfactor_modes(sbdry_init,coords,modes,frequencies_plot, prop_matrix_plot,
                                   diskR=diskR, prop=prop)

#Only plot fundamental mode
plot(frequencies_plot.*1e-9,abs2.(Eout_init)[1,Lmax+1,:])
xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")

# %% [markdown]
# # Propagation Matrix Interpolation

# %%
#Calculate interpolation object for frequencies we want to optimize. To avoid calculating the
#costly propagation matrix for each possible distance, we calculate it only at some points and
#interpolate the others. The spacing grid is relative distance changes from the provided sbdry
#struct.  The grid steps shouldn't be larger than 50 um. Choose a sensible range from to safe
#computation (e.g. a small range if you are already close to the optimum, larger if far away) 
spacing_grid = (-2500:50:2500)*1e-6;

prop_matrix_grid_sub(center) = calc_propagation_matrices_grid(sbdry_init, coords, modes,
                                    spacing_grid, frequencies_optim(center, freq_width),
                                    diskR=diskR, prop=prop);
#This is the interpolation object that is being called when interpolating
itp_sub(center) = construct_prop_matrix_interpolation(prop_matrix_grid_sub(center),
                                                      spacing_grid)

# %% [markdown]
# # Optimization

# %%
spacings = zeros(n_disk)
#We define a single argument cost function for the optimizer
#m_reflect defines the combination of modes we want to optimize. For now we are only interested in the fundamental mode
function cost_fun(fixed_disk, center)
    disk = copy(fixed_disk)
    itp = itp_sub(center)
    freq_o = frequencies_optim(center, freq_width)
    return x -> begin
        # if one disk is fixed, insert it into dist_shift as the optimizer then runs on one
        # dimension less but we still need the "full" list of disks
        if disk > 0
            # it's important to *copy* and not modify x here otherwise the optimizer gets confused
            x = vcat(x[1:disk - 1], spacings[disk], x[disk:length(x)])
        end

        calc_boostfactor_cost(x, itp, freq_o, sbdry_init, coords, modes, m_reflect,
                              diskR=diskR, prop=prop)
    end
end;

# %%
# With NelderMead. Slow but robust
#algorithm = NelderMead()
#options = Optim.Options(iterations=1000000)

#Quasi-Newton (gradient-based). Fast but likes to get stuck in local minima. 
algorithm = LBFGS(linesearch = BackTracking(order=2))
options = Optim.Options(f_tol = 1e-6)

# %%
GC.gc()

#spacings = [-0.000247484690809835,
#             5.494161169410954e-5,
#             1.4082408393016912e-5,
#             3.474533260803849e-6,
#             1.4127821024835357e-6,
#            -7.062735634745444e-6,
#            -5.658759242525849e-5,
#            -0.0001736356257557033,
#             0.0003140458044571532,
#             0.000946244385671497,
#             6.889677168591535e-6,
#            -0.00015884778788364403,
#            -0.0001663066683797041,
#            -0.00035282832642484133,
#             0.0008533180045262031,
#            -0.002345992812444579,
#             0.0009458750290604845,
#            -0.0004042461326743376,
#            -0.00043332980022113657,
#             0.0004518582202935378]

function optimize_spacings(cost_function; starting_point=zeros(n_disk))
    spacings = Vector{Float64}()
    best_cost = Atomic{Float64}(1000.)
    lk = SpinLock()
    # Run initial optimization a few times and pick the best one
    @threads for i in 1:100
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

# spacings = optimize_spacings(cost_fun(0, freq_center))

spacings = readdlm("optim_results/22.288GHz-2021-04-14T13:13.txt", ',')[:]
shift = -150 # Shift in MHz
results = Vector{Vector{Float64}}(undef, n_disk + 1)

# %%
for i in 0:n_disk
    println("Optimizing for fixed disk $i")
    n_moving_disks = n_disk - (i > 0 ? 1 : 0)
    x_0 = vcat(spacings[1:i-1], spacings[i+1:n_disk])
    results[i + 1] = optimize_spacings(cost_fun(i, freq_center + shift * 1e6), starting_point=x_0)
end

# %%
function calc_eout(distances; fixed_disk=0)
    dist = copy(distances)
    if fixed_disk > 0
        insert!(dist, fixed_disk, spacings[fixed_disk])
    end
    sbdry_optim = copy_setup_boundaries(sbdry_init, coords)
    sbdry_optim.distance[2:2:end-2] .+= dist

    #Calculate prop matrix grid at a dist shift of zero of optimized setup
    prop_matrix_grid_plot = calc_propagation_matrices_grid(sbdry_optim, coords,modes, 0,
                                                           frequencies_plot, diskR=diskR,
                                                           prop=prop)
    prop_matrix_plot = [prop_matrix_grid_plot[r,f,1,1,1,:,:] for r = 1:n_region,
                        f = 1:length(frequencies_plot)]
    calc_boostfactor_modes(sbdry_optim, coords, modes, frequencies_plot,
                                           prop_matrix_plot, diskR=diskR, prop=prop)
end
#Plot results
#Create optimized setup
GC.gc()
Eout_0 = calc_eout(spacings)

#results = readdlm("optim_results/22.288GHz+30MHz-2021-04-14T18:01.txt", ',')
#results = [filter(x -> x != "", results[i,:]) for i in 1:size(results, 1)]
Eout_optim = Array{Array{ComplexF64, 3}}(undef, n_disk + 1)
@time @threads for (i, res) in collect(enumerate(results))
    Eout_optim[i] = calc_eout(res, fixed_disk=i - 1)
end

# %%
function write_to_file(data, name)
    date = Dates.format(now(), "yyyy-mm-ddTHH:MM")
    open("optim_results/$name-$date.txt", "w") do io
        writedlm(io, data, ',')
    end
end

#write_to_file(spacings, "$(freq_center * 1e-9)GHz")
write_to_file(results, "$(freq_center * 1e-9)GHz+$(shift)MHz")

# %%
v = Dates.format(now(), "yyyy-mm-ddTHH:MM")

# %%
clf()
legend_text = Array{String}(undef, 0)
push!(legend_text, "freq_0")
plot(frequencies_plot .* 1e-9, abs2.(Eout_0[1, Lmax + 1, :]))
for i in 1:6
    out = Eout_optim[i]
    plot(frequencies_plot.*1e-9,abs2.(out[1, Lmax + 1, :]))
    push!(legend_text, "Fixed disk: $(i-1)")
end
xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")
legend(legend_text)
savefig("$(shift)MHz_1_$v", dpi=480)

# %%
clf()
legend_text = Array{String}(undef, 0)
push!(legend_text, "freq_0")
plot(frequencies_plot .* 1e-9, abs2.(Eout_0[1, Lmax + 1, :]))
for i in vcat(1, 7:11)
    out = Eout_optim[i]
    plot(frequencies_plot.*1e-9,abs2.(out)[1,Lmax+1,:])
    push!(legend_text, "Fixed disk: $(i-1)")
end
xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")
legend(legend_text)
savefig("$(shift)MHz_2_$v", dpi=480)

# %%
clf()
legend_text = Array{String}(undef, 0)
push!(legend_text, "freq_0")
plot(frequencies_plot .* 1e-9, abs2.(Eout_0[1, Lmax + 1, :]))
for i in vcat(1, 12:16)
    out = Eout_optim[i]
    plot(frequencies_plot.*1e-9,abs2.(out)[1,Lmax+1,:])
    push!(legend_text, "Fixed disk: $(i-1)")
end
xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")
legend(legend_text)
savefig("$(shift)MHz_3_$v", dpi=480)

# %%
clf()
legend_text = Array{String}(undef, 0)
push!(legend_text, "freq_0")
plot(frequencies_plot .* 1e-9, abs2.(Eout_0[1, Lmax + 1, :]))
for i in vcat(1, 17:21)
    out = Eout_optim[i]
    plot(frequencies_plot.*1e-9,abs2.(out)[1,Lmax+1,:])
    push!(legend_text, "Fixed disk: $(i-1)")
end
xlabel("Frequency [GHz]")
ylabel("Power Boostfactor")
legend(legend_text)
savefig("$(shift)MHz_4_$v", dpi=480)

# %%
