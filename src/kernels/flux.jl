using StaticArrays
using KernelAbstractions

"""
    evaluate_fluxes(eq::Equation, grid::Grid, dofs)

Evaluates the fluxes for the given equation `eq` on the grid `grid` with the
given degrees of freedom `dofs`. Updates `grid.flux` in place.
Possibly runs on a GPU.
"""
function evaluate_fluxes(eq::TerraDG.Equation, grid::TerraDG.Grid{F,T1,T2,2,ndofs,order,T3},dofs::T2) where {F<:Real,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    cells = grid.cells
    flux = grid.flux
    backend = get_backend(grid.flux)
    @kernel function evaluate_cell_flux()
        cell_id = @index(Global)
        cellflux = @view flux[cell_id, 1:order^2 * 2, 1:ndofs]
        celldofs = SMatrix{order^2,ndofs,F}(@view dofs[cell_id, 1:order^2, 1:ndofs])
        evaluate_flux(eq, celldofs, cellflux)
    end
    kernel = evaluate_cell_flux(backend)
    kernel(ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end