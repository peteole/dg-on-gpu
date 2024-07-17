using Revise
using TerraDG
using YAML
using BenchmarkTools
include("benchmarkutils.jl")

for base_config in readdir("benchmark/configs")
    for device in [cpu,cuda]
        config = TerraDG.Configuration("benchmark/configs/$base_config")
        config = device(config)
        @info "Running benchmarks for $base_config on device $(config.device)"
        results = run_benchmarks(config)
        result_path = get_benchmark_result_path(config)
        # make sure the directory exists
        mkpath(dirname(result_path))
        YAML.write_file(result_path, results)
    end
end