using Revise
using TerraDG
using YAML
using BenchmarkTools
include("benchmarkutils.jl")

function run(filename::String)
config = TerraDG.Configuration(filename)
grid_elements=400
for device in [cuda, cpu]
    for order in [1,2,3,4,5,6,7,8]
        config = set_grid_size(config, grid_elements)
        config = set_order(config, order)
        config = device(config)
        @info "Running benchmarks for $config on device $(config.device)"
        results = run_benchmarks(config)
        result_path = get_benchmark_result_path(config)
        # make sure the directory exists
        mkpath(dirname(result_path))
        YAML.write_file(result_path, results)
    end
end
end
run("benchmark/configs/euler.yaml")