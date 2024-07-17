using TerraDG
using BenchmarkTools
using Statistics

function create_setup(config::TerraDG.Configuration)
    # remove old output
    rm("output", recursive=true, force=true)
    mkdir("output")
    eq = TerraDG.make_equation(config)
    scenario = TerraDG.make_scenario(config)
    array_type=TerraDG.get_device_array_type(config)
    grid = TerraDG.make_grid(config, eq, scenario, array_type, typeof(config.end_time))
    F=typeof(grid.time)
    integrator = TerraDG.make_timeintegrator(config, grid)
    globals = TerraDG.GlobalMatrices(grid.basis, grid, grid.basis.dimensions)

    filename = "output/plot"

    plotter = TerraDG.VTKPlotter(eq, scenario, grid, filename, config.save_images)
    return eq, scenario, globals, grid, plotter, integrator
end

function cpu(config::TerraDG.Configuration)
    config_copy=deepcopy(config)
    config_copy.device="cpu"
    config_copy
end

function cuda(config::TerraDG.Configuration)
    config_copy=deepcopy(config)
    config_copy.device="cuda"
    config_copy
end

function set_grid_size(config::TerraDG.Configuration, s::Int)
    config_copy=deepcopy(config)
    config_copy.grid_elements=s
    config_copy
end

function set_order(config::TerraDG.Configuration, s::Int)
    config_copy=deepcopy(config)
    config_copy.order=s
    config_copy
end

function create_setup(configfile::String)
    config = TerraDG.Configuration(configfile)
    return create_setup(config)
end

function get_benchmark_result_path(config::TerraDG.Configuration)
    nthreads=Threads.nthreads()
    return "benchmark/results/eq=$(config.equation_name),scenario=$(config.scenario_name),device=$(config.device),size=$(config.grid_elements),nthreads=$(nthreads),order=$(config.order).yaml"
end

function benchmark_result_to_dict(result::BenchmarkTools.Trial)
    times=result.times .* 1e-9 # convert to seconds
    return Dict(
        "time" => Dict(
            "mean"=> mean(times),
            "std" => std(times),
            "min" => minimum(times),
            "max" => maximum(times),
            "median" => median(times),
        ),
        "memory" => result.memory, # bytes
        "nallocs" => result.allocs,
    )
end


function run_step(integrator, grid::TerraDG.Grid{F,T1,T2,ndims,ndofs,order,T3}, eq, scenario, globals, du, dofs) where {F,T1,T2,ndims,ndofs,order,T3}
    TerraDG.step(integrator, grid, F(0)) do du, dofs, time
        TerraDG.evaluate_rhs(eq, scenario, globals, du, dofs, grid)
    end
end

function run_benchmarks(config::TerraDG.Configuration)
    eq, scenario, globals, grid, plotter, integrator = create_setup(config)
    du = similar(grid.dofs)
    dofs = grid.dofs

    @info "benchmarking initial conditions"
    b_initial_conditions = @benchmark TerraDG.set_initial_conditions($eq, $scenario, $grid)
    display(b_initial_conditions)
    @info "benchmarking timestep"
    b_timestep = @benchmark run_step($integrator, $grid, $eq, $scenario, $globals, $du, $dofs)
    display(b_timestep)
    @info "benchmarking evaluate_fluxes"
    b_evaluate_fluxes = @benchmark TerraDG.evaluate_fluxes($eq, $grid, $dofs)
    display(b_evaluate_fluxes)
    @info "benchmarking evaluate_dofs"
    b_evaluate_dofs = @benchmark TerraDG.evaluate_dof_points($grid, $plotter.cellpoints_gpu, 1)
    display(b_evaluate_dofs)
    @info "benchmarking evaluate_rhs"
    b_rhs = @benchmark TerraDG.evaluate_rhs($eq, $scenario, $globals, $du, $dofs, $grid)
    display(b_rhs)
    @info "benchmarking evaluate_volumes"
    b_evaluate_volumes = @benchmark TerraDG.evaluate_volumes($globals, $grid, $du)
    display(b_evaluate_volumes)
    @info "benchmarking evaluate_face_integrals"
    b_evaluate_face_integrals = @benchmark TerraDG.evaluate_face_integrals($eq, $scenario, $globals, $grid, $du, $dofs)
    display(b_evaluate_face_integrals)


    Dict(
        "evaluate_rhs" => benchmark_result_to_dict(b_rhs),
        "evaluate_fluxes" => benchmark_result_to_dict(b_evaluate_fluxes),
        "evaluate_volumes" => benchmark_result_to_dict(b_evaluate_volumes),
        "evaluate_face_integrals" => benchmark_result_to_dict(b_evaluate_face_integrals),
        "set_initial_conditions" => benchmark_result_to_dict(b_initial_conditions),
        "evaluate_dofs" => benchmark_result_to_dict(b_evaluate_dofs),
        "timestep" => benchmark_result_to_dict(b_timestep),
    )
end