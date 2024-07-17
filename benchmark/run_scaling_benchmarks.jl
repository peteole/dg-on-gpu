using Revise
using TerraDG
using YAML
using BenchmarkTools
include("benchmarkutils.jl")

function run(filename::String)
    config = TerraDG.Configuration(filename)
    for device in [cuda, cpu]
        for grid_elements in [400, 25, 50, 100, 200, 800, 1600]
            config = set_grid_size(config, grid_elements)
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