module Thesis

include("common/FileUtils.jl")
include("common/transformer_optim_utilities.jl")
include("common/BoostFactorOptimizer.jl")

include("eigendirections/eigendirections.jl")
include("eigendirections/plot_stuff.jl")

#include("fixed_disks/plot_stuff.jl")

include("ml/neuralnet.jl")

for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end # module
