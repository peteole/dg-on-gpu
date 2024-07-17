module TerraDG
using WriteVTK
using Printf
using Logging
using LinearAlgebra
import YAML
using ProgressMeter
using Glob

include("configuration.jl")
include("basis.jl")
include("equations.jl")
include("grid.jl")
include("global_matrices.jl")
include("kernels/flux.jl")
include("kernels/surface.jl")
include("kernels/volume.jl")
include("kernels/time.jl")
include("plotters.jl")
include("error_writer.jl")
include("kernels/slope_limiting.jl")
include("kernels/initial_condition.jl")
include("kernels/eigenvalue.jl")
global_logger(ConsoleLogger(stderr, Logging.Warn))
"""
    evaluate_rhs(eq, scenario, filter, globals, du, dofs, grid)

Evalutes the right-hand-side of the equation `eq` for 
scenario `scenario`, with filter `filter`, 
collection of global matrices `globals`, update
`du`, degrees of freedom `dofs` and grid `grid`.

Updates `du` in place.
"""
function evaluate_rhs(eq, scenario, globals, du, dofs, grid)
    du .= 0
    evaluate_fluxes(eq, grid, dofs)
    evaluate_volumes(globals, grid, du)
    evaluate_face_integrals(eq, scenario, globals, grid, du, dofs)
end

main(configfile::String) = main(Configuration(configfile))

function setup_simulation(config::Configuration)
    eq = make_equation(config)
    scenario = make_scenario(config)
    array_type = get_device_array_type(config)
    grid = make_grid(config, eq, scenario, array_type, typeof(config.end_time))
    F = typeof(grid.time)
    slope_limiting_buffer = SlopeLimitingBuffer(grid, array_type, F)
    integrator = make_timeintegrator(config, grid)
    globals = GlobalMatrices(grid.basis, grid, grid.basis.dimensions)
    return eq, scenario, grid, slope_limiting_buffer, integrator, globals, F
end

"""
    main(configfile::String)

Runs a DG-simulation with configuration from `configfile`.
"""
function main(config::Configuration)
    # remove old output
    rm.(glob("output/*"), force=true, recursive=true)
    rm.(glob("plots/**"), force=true, recursive=true)
    mkpath("output")
    mkpath("plots")
    eq, scenario, grid, slope_limiting_buffer, integrator, globals, F = setup_simulation(config)
    @info "Using" F
    @info "Initialised grid"
    @info "Order" config.order
    @info "Courant" config.courant
    @info "Device" config.device

    filename = "output/plot"

    # Init everything
    set_initial_conditions(eq, scenario, grid)
    plotter = VTKPlotter(eq, scenario, grid, filename, config.save_images)

    grid.time = 0
    timestep = 0
    next_plotted = config.plot_start

    # Worksheet 2 - only do interpolation, plotting and error
    if typeof(scenario) == Interpolation
        plot(plotter, grid)
        save(plotter)
        evaluate_error(eq, scenario, grid, grid.time)
        return
    end
    pbar = Progress(100;desc="Time stepping")
    while grid.time < config.end_time
        if timestep > 0
            time_start = time()
            dt::F = F(1 / (config.order^2 + 1) * config.cellsize[1] * config.courant * 1 / grid.maxeigenval)
            # Only step up to either end or next plotting
            dt = min(dt, next_plotted - grid.time, config.end_time - grid.time)

            @assert dt > 0 "Negative timestep: $dt"

            if config.slope_limiting
                compute_slope_limiting_coefficients!(grid, slope_limiting_buffer)
                minmod_slope_limiting!(grid, slope_limiting_buffer)
            end
            @info "Running timestep" timestep dt grid.time
            step(integrator, grid, dt) do du, dofs, time
                evaluate_rhs(eq, scenario, globals, du, dofs, grid)
            end

            grid.time += dt
            time_end = time()
            time_elapsed = time_end - time_start
            @info "Timestep took" time_elapsed
        else
            update_max_eigenval!(grid, eq)
            @info "Set initial max eigenvalue" grid.maxeigenval
        end

        if abs(grid.time - next_plotted) < 1e-5
            @info "Writing output" grid.time
            plot(plotter, grid)
            next_plotted = grid.time + config.plot_step
            update!(pbar, floor(Int, grid.time/config.end_time*100))
        end
        timestep += 1
    end
    save(plotter)
    if is_analytical_solution(eq, scenario)
        evaluate_error(eq, scenario, grid, grid.time)
        #evaluate_error(eq, scenario, grid, grid.time)
    end
end

end
