using KernelAbstractions

function update_max_eigenval!(grid::Grid{F,T1,T2,2,ndofs,order,T3},eq::Equation) where {F,T1,T2,ndofs,order,T3}
    dofs=grid.dofs

    @kernel function compute_eigenvalues(eigenvals)
        cell_id=@index(Global)
        @views dofscell = dofs[cell_id, 1:order^2,1:ndofs]
        celldata = SMatrix{order^2,ndofs,F}(dofscell)
        eigenvals[cell_id] = max(max_eigenval(eq, celldata, 1), max_eigenval(eq, celldata, 2))
    end
    backend=get_backend(grid.eigenvals)
    @info "backend" backend
    kernel=compute_eigenvalues(backend)
    kernel(grid.eigenvals,ndrange=length(grid.cells))
    KernelAbstractions.synchronize(backend)
    grid.maxeigenval = maximum(grid.eigenvals[1:length(grid.cells)])
end