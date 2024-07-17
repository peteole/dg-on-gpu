# COV_EXCL_START
using StaticArrays
using LinearAlgebra
using KernelAbstractions

"""
    evaluate_error(eq, scenario, grid, t)

Evaluates the error of the solution for the equation `eq` and scenario `scenario` on the grid `grid` at time `t`. Writes the error to the console.
Evaluates the error in the L1, L2 and L∞ norms.
"""
function evaluate_error(eq::Equation, scenario::Scenario, grid::Grid{F,T1,T2,2,ndofs,order,T3}, t::Real) where {F,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    @assert is_analytical_solution(eq, scenario)
    @info "Evaluating error for $(typeof(eq))"

    cells = grid.cells
    ncells = length(cells)

    l1_errors = similar(grid.dofs, F, ncells, ndofs)
    l2_errors = similar(grid.dofs, F, ncells, ndofs)
    linf_errors = similar(grid.dofs, F, ncells, ndofs)
    l1_errors .= 0
    l2_errors .= 0
    linf_errors .= 0

    l1_error = zeros(Float64, ndofs)
    l2_error = zeros(Float64, ndofs)
    linf_error = zeros(Float64, ndofs)

    grid_dofs = grid.dofs
    basispoints_1d = grid.basis.quadpoints

    quad_order = max(order + 2, 6)
    quad_basis = Basis(quad_order, grid.basis.dimensions)
    quadpoints_1d = quad_basis.quadpoints
    quadweights_1d = quad_basis.quadweights

    backend = get_backend(cells)

    # compute the error cell-wise
    @kernel function evaluate_error_kernel(l1_errors, l2_errors, linf_errors)
        cell_id = @index(Global)
        cell = cells[cell_id]

        for i = 1:quad_order
            for j = 1:quad_order
                x = quadpoints_1d[i]
                y = quadpoints_1d[j]

                quadweight = quadweights_1d[i] * quadweights_1d[j]
                weight = volume(cell) * quadweight

                pxg, pyg = globalposition(cell, (x, y))
                analytical = get_initial_values(eq, scenario, (pxg, pyg), t=t)
                for var = 1:ndofs
                    dofcell = @view grid_dofs[cell_id, :, var]

                    cartesian_idx = CartesianIndices((order, order))
                    value = 0
                    for linear_idx in 1:order^2
                        value += dofcell[linear_idx] * lagrange_1d_gpu(order, basispoints_1d, cartesian_idx[linear_idx][1], x) * lagrange_1d_gpu(order, basispoints_1d, cartesian_idx[linear_idx][2], y)
                    end

                    error = abs(value - analytical[var])

                    l1_errors[cell_id, var] += weight * error
                    l2_errors[cell_id, var] += weight * error^2
                    linf_errors[cell_id, var] = max(linf_errors[cell_id, var], error)
                end
            end
        end
    end

    kernel = evaluate_error_kernel(backend)
    kernel(l1_errors, l2_errors, linf_errors, ndrange=ncells)
    KernelAbstractions.synchronize(backend)

    # accumulate results
    for var = 1:ndofs
        l1_error[var] = sum(l1_errors[:, var])
        l2_error[var] = sum(l2_errors[:, var])
        linf_error[var] = maximum(linf_errors[:, var])
    end

    l2_error .= sqrt.(l2_error)
    println("Errors for each variable")
    println("Var\tL1\tL2\tL∞")
    for i = 1:ndofs
        var_name = string(get_variable_name(eq, i))
        @printf "%s\t%15.6e\t%15.6e\t%15.6e\n" var_name l1_error[i] l2_error[i] linf_error[i]
    end

end
# COV_EXCL_STOP
