using DelimitedFiles
using Dates

#prefix = "/beegfs/desy/user/lepplada/"
prefix = ""

function get_current_version(file)
    i = 1
    while isfile("$(file)_v$i.txt")
        i += 1
    end
    return "v$i"
end

function write_init_spacing_to_file(data, freq)
    open("$(prefix)results/init_$freq.txt", "w") do io
        write(io, "$data")
    end
end

function write_optim_spacing_to_file(data, freq)
    v = get_current_version("$(prefix)results/optim_$(freq)")
    open("$(prefix)results/optim_$(freq)_$v.txt", "w") do io
        writedlm(io, data, ',')
    end
end

function read_init_spacing_from_file(file)
    open(file, "r") do io
        parse(Float64, read(io, String))
    end
end

function read_optim_spacing_from_file(file)
    open("$prefix$file", "r") do io
        readdlm(io, ',')[:]
    end
end
