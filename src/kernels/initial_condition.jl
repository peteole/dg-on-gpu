using KernelAbstractions


"""
    set_initial_conditions(eq::Equation, scenario::Scenario, grid::Grid)

Sets the initial conditions for the given equation `eq` and scenario `scenario`
on the grid `grid`. Possibly runs on a GPU.
"""
function set_initial_conditions(eq::Equation, scenario::Scenario, grid::TerraDG.Grid{F,T1,T2,2,ndofs,order,T3}) where {F<:Real,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    cells = grid.cells
    basis = grid.basis
    quadpoints_1d = basis.quadpoints
    backend = get_backend(grid.dofs)
    @kernel function set_initial_condition(dofs)
        cell_id = @index(Global)
        celldofs = @view dofs[cell_id, :, :]
        cell = cells[cell_id]
        ci = CartesianIndices((order, order))
        for i in 1:order^2
            cell_pos = (quadpoints_1d[ci[i][1]], quadpoints_1d[ci[i][2]])
            celldofs[i, :] .= get_initial_values(eq, scenario, globalposition(cell, cell_pos))
        end
    end
    kernel = set_initial_condition(backend)
    kernel(grid.dofs, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end
