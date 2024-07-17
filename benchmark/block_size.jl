using Revise
using TerraDG
using CUDA
include("benchmarkutils.jl")
config=TerraDG.Configuration("benchmark/configs/advection.yaml") |> cuda
eq, scenario, filter, globals, grid, plotter, integrator,slope_limiting_buffer = create_setup(config)
du = similar(grid.dofs)
dofs = grid.dofs
TerraDG.set_initial_conditions(eq, scenario, grid)
CUDA.@profile TerraDG.evaluate_face_integrals(eq, scenario, globals, grid, du, dofs)
