using DelimitedFiles
using Dates

function write_init_spacing_to_file(data, freq)
    open("results/init_$freq.txt", "w") do io
        write(io, "$data")
    end
end

function write_optim_spacing_to_file(data, freq)
    date = Dates.format(now(), "yyyy-mm-ddTHH:MM")
    open("results/optim_$(freq)_$date.txt", "w") do io
        writedlm(io, data, ',')
    end
end

function read_init_spacing_from_file(file)
    open(file, "r") do io
        parse(Float64, read(io, String))
    end
end

function read_optim_spacing_from_file(file)
    open(file, "r") do io
        readdlm(io, ',')[:]
    end
end
