using Base: @kwdef
using BSON: @save, @load
using CUDA
using Distributions
using Flux: @epochs
using Flux, Statistics
using JSON
using Plots
using Random


VARIATION=0.5e-3
freq_center = 20e9
freq_range = (freq_center - 5e9):0.04e9:(freq_center + 5e9)

function gen_spacing(n_disk)
    thickness = 1e-3
    length = 304e-3 - thickness
    min_dist = 6e-3 + thickness
    x_0 = Vector{Float64}(undef, n_disk)
    for i in 1:n_disk
        x_0[i] = rand(Uniform(i == 1 ? 0. : min_dist, length - (n_disk - i) * min_dist), 1)[1]
    end
    x_0
end

function gen_data(n; n_disk=20)#diff=1e-3, offset=0)
    n_disk = 2#20
    n_region = 2 * n_disk + 2
    epsilon = 24 * (1 - 1e-3im)

    eps = vcat(1e20, reduce(vcat, [1, epsilon] for i in 1:n_disk), 1)
    init_spacings = 0.

    distances = distances_from_spacing(init_spacings, n_region)

    optim_params = init_optimizer(n_disk, epsilon, 0.15, 1, 0, freq_center, 50e6, freq_range,
                                  distances, eps)
    data = Array{Tuple{Array{Float64}, Array{Float64}}, 1}(undef, n)
    for i in 1:n
        #x_0 = [diff * i + offset] #zeros(n_disk) .+ 2 .* (rand(n_disk).-0.5) .* VARIATION
        x_0 = gen_spacing(n_disk)
        eout = calc_eout(optim_params, x_0, reflect=true, disable_constraints=false)
        eout[2, 1, :] .*= [exp(-1im * sum(x_0) / 3e8 * 2pi * f) for f in optim_params.freq_range]
        data[i] = (reduce(vcat, map(x -> [real(x), imag(x)], log.(eout[2, 1, :]))),
                   reduce(vcat, map(x -> abs2(x), eout[1, 1, :])))
    end
    data
end

function save_dataset(file, n; kwargs...)
    data = gen_data(n; kwargs...)

    open(file, "w") do f
        JSON.print(f, data)
    end
end

function load_dataset(file)
    data = Vector{Tuple{Array{Float64}, Array{Float64}}}()
    for d in JSON.parsefile(file)
        push!(data, convert(Tuple{Array{Float64}, Array{Float64}}, (d[1], d[2])))
    end
    data
end

function getdata()
    train = gen_data(5000)
    test = gen_data(1000)

    return train, test#DataLoader(train[1], batchsize=1), DataLoader(test[1], batchsize=1)
end

function build_model()
    return Chain(Dense(length(freq_range) * 2, 256),
                 Dropout(0.3),
                 x -> reshape(x, :, 1, 1),
                 Conv((3, ), 1 => 8, pad=(1, ), σ),
                 x -> maxpool(x, (2, )),
                 Conv((3, ), 8 => 16, pad=(1, ), σ),
                 x -> maxpool(x, (2, )),
                 Conv((3, ), 16 => 16, pad=(1, ), σ),
                 x -> maxpool(x, (2, )),
                 x -> reshape(x, :, size(x, 3)),
                 #x -> begin
                 #    println("x: $(length(x)), l: $(length(optim_params.freq_range) * 4)")
                 #    return x
                 #end,
                 Dense(#length(optim_params.freq_range) * 4,
                       #992,
                       256 * convert(Int, 16 / 2^3),
                       length(freq_range)))
                 #Dense(length(optim_params.freq_range) * 2, 256),
                 #Dropout(0.2),
                 #Dense(256, 128, σ),
                 #Dense(128, 64, σ),
                 #Dense(64, length(optim_params.freq_range)))
end

function loss_and_accuracy(data_loader, model)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        ls += Flux.mse(model(x), y, agg=sum)
        acc += sum(abs2.(model(x) - y))
        num +=  1
    end
    return ls / num, acc / num
end

function train(train_file, test_file, epochs)

    model = build_model() |> gpu

    #@load "test_model.bson" model

    ps = Flux.params(model)

    opt = ADAM(1.e-4)

    train_loader = load_dataset(train_file) |> gpu
    test_loader = load_dataset(test_file) |> gpu

    println("Got data, starting training…")
    sleep(1)

    loss_vector = Vector{Tuple{Float64, Float64}}()
    loss_vector_test = Vector{Tuple{Float64, Float64}}()

    for epoch in 1:epochs
        for (x, y) in shuffle(train_loader)
            gradient = Flux.gradient(ps) do
                training_loss = Flux.mse(model(x), y)
                return training_loss
            end

            Flux.update!(opt, ps, gradient)
        end

        ls, acc = loss_and_accuracy(train_loader, model)
        ls2, acc2 = loss_and_accuracy(test_loader, model)
        println("Epoch: $epoch\nLoss: $ls\tTest loss: $acc2\n")
        push!(loss_vector, (ls, acc))
        push!(loss_vector_test, (ls2, acc2))
    end

    test_ls, test_acc = loss_and_accuracy(test_loader, model)
    #train_ls, train_acc = loss_and_accuracy(train_loader, model)
    println("Test dataset - Loss: $test_ls, Acc: $test_acc")
    #println("Train dataset - Loss: $train_ls, Acc: $train_acc")

    @save "test_model.bson" model
    p1 = plot(1:length(loss_vector), map(x->x[1], loss_vector), label="Train")
    plot!(p1, 1:length(loss_vector_test), map(x->x[1], loss_vector_test), label="Test")

    p2 = plot(optim_params.freq_range ./ 1e9, test_loader[10][2] |> cpu)
    plot!(p2, optim_params.freq_range ./ 1e9, model(test_loader[10][1]) |> cpu)

    map(display, [p1, p2])
end

to_complex(x) = begin
    res = Vector{Complex}(undef, div(length(x), 2))
    for i in 1:2:length(x)
        res[div((i-1), 2) + 1] = Complex(x[i], x[i+1])
    end
    res
end

function predict(freqs)
    model = build_model()
    @load "test_model.bson" model

    model(freqs |> gpu)
end
