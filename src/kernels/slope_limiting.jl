"""
    SlopeLimitingBuffer
A buffer to store the cell averages and slopes for slope limiting.
    All fields are potentially on GPU.
"""
struct SlopeLimitingBuffer{F, T1 <: AbstractArray{F,2}, T2 <: AbstractArray{F,3}}
    cell_averages::T1 # cell_idx x dof_idx
    cell_slopes::T2 # cell_idx x dof_idx x direction
    function SlopeLimitingBuffer(grid::Grid, ::Type{T}=Array, ::Type{F}=Float64) where {F, T<:AbstractArray}
        new{F, T{F,2}, T{F,3}}(T{F, 2}(undef, (length(grid.cells), size(grid.dofs, 3))),T{F, 3}(undef,(length(grid.cells), size(grid.dofs, 3), 2)))
    end
end

"""
    compute_slope_limiting_coefficients!(grid::Grid, buffer::SlopeLimitingBuffer)

Compute the cell averages and slopes for slope limiting on the grid `grid` and store them in `buffer`.
Potentially runs on a GPU.
Slopes and averages are estimated using Gauss quadrature.
"""
function compute_slope_limiting_coefficients!(grid::Grid{F,T1,T2,2,ndofs,order,T3}, buffer::SlopeLimitingBuffer) where {F,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    quadweights_1d = grid.basis.quadweights
    quadweights= SVector{order^2}(quadweights_nd(grid.basis))
    quadpoints = SVector{order}(grid.basis.quadpoints)

    cells = grid.cells
    ncells = length(cells)
    dofs = grid.dofs

    cell_slopes = buffer.cell_slopes
    cell_averages = buffer.cell_averages

    @kernel function evaluate_slope_limiting_coeffs_kernel()
        cell_id = @index(Global)
        celldata = @view dofs[cell_id, 1:order^2, 1:ndofs]
        for dofidx in 1:ndofs
            celldof = SVector{order^2}(@view celldata[1:order^2, dofidx])
            cell_averages[cell_id, dofidx] = sum(celldof .* quadweights)
            cell_slopes[cell_id, dofidx, 1] = 0
            cell_slopes[cell_id, dofidx, 2] = 0
            linear_grid_index = LinearIndices((order, order))
            # Perform a Gauss quadrature to estimate the slopes, summing over all grid points
            for basis_point in CartesianIndices((order, order))
                i, j = Tuple(basis_point)
                x, y = quadpoints[i], quadpoints[j]
                lin_idx = linear_grid_index[basis_point]
                cell_slopes[cell_id, dofidx, 1] += 12 * ((x - F(1) / 2) * celldata[lin_idx, dofidx]) * quadweights_1d[i] * quadweights_1d[j]
                cell_slopes[cell_id, dofidx, 2] += 12 * ((y - F(1) / 2) * celldata[lin_idx, dofidx]) * quadweights_1d[i] * quadweights_1d[j]
            end
        end
    end

    backend = get_backend(cells)
    kernel = evaluate_slope_limiting_coeffs_kernel(backend)
    kernel(ndrange=ncells)
    KernelAbstractions.synchronize(backend)
end

function minmod(s1, s2, s3)
    condition = ((sign(s1 * s2) + 1 ) / 2) * ((sign(s1 * s3) + 1 ) / 2)
    return sign(s1) * min(abs(s1), abs(s2), abs(s3)) * condition
end

"""
    minmod_slope_limiting!(grid::Grid, buffer::SlopeLimitingBuffer)

Perform slope minmod limiting on the grid `grid` using the cell averages and slopes stored in `buffer`.
Potentially runs on a GPU.
"""
function minmod_slope_limiting!(grid::Grid{F,T1,T2,2,ndofs,order,T3}, buffer::SlopeLimitingBuffer) where {F,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    quadpoints = grid.basis.quadpoints
    basissize_1d = length(quadpoints)

    quadpoints = grid.basis.quadpoints
    quadpoints_svec = SVector{order, F}(quadpoints...)

    cells = grid.cells
    dofs = grid.dofs
    ncells = length(cells)

    cell_slopes = buffer.cell_slopes
    cell_averages = buffer.cell_averages

    # very technical, follows the lecture notes
    @kernel function evaluate_minmod_slope_limiting_kernel()
        cell_id = @index(Global)
        cell = cells[cell_id]
        dx = 1
        dy = 1

        linear_grid_index = LinearIndices((order, order))
        for dofidx in 1:ndofs
            a = cell_averages[cell_id, dofidx]
            b = cell_slopes[cell_id, dofidx, 1]
            c = cell_slopes[cell_id, dofidx, 2]

            u_o = cell_averages[cell_id, dofidx]


            u_l = cell_averages[cell.neighbor_indices[Int(left)], dofidx]
            u_r = cell_averages[cell.neighbor_indices[Int(right)], dofidx]

            s1x = b / dx
            s2x = cell.facetypes[Int(left)] == boundary ? s1x : (u_o - u_l) / dx
            s3x = cell.facetypes[Int(right)] == boundary ? s1x : (u_r - u_o) / dx
            s_star_x = minmod(s1x, s2x, s3x)

            u_t = cell_averages[cell.neighbor_indices[Int(top)], dofidx]
            u_b = cell_averages[cell.neighbor_indices[Int(bottom)], dofidx]
            s1y = c / dy
            s2y = cell.facetypes[Int(bottom)] == boundary ? s1y : (u_o - u_b) / dy
            s3y =  cell.facetypes[Int(top)] == boundary ? s1y : (u_t - u_o) / dy
            s_star_y = minmod(s1y, s2y, s3y)

            n_x = abs(s1x)
            m_x = abs(s_star_x)

            n_y = abs(s1y)
            m_y = abs(s_star_y)
            
            condition = max((sign(n_x - m_x) + 1) / 2, (sign(n_y - m_y) + 1) / 2) # = ((abs(s1x) > abs(s_star_x)) ||  (abs(s1y) > abs(s_star_y)))
            
            for basis in CartesianIndices((order, order))
                i, j = Tuple(basis)
                x, y = quadpoints_svec[i], quadpoints_svec[j]

                orig_val = dofs[cell_id, linear_grid_index[basis], dofidx]
                new_val = (a + s_star_x * (x - 1/2) + s_star_y * (y - 1/2))

                updateval = condition * new_val + (1 - condition) * orig_val
                dofs[cell_id, linear_grid_index[basis], dofidx] = updateval
            end
        end
    end
    backend = get_backend(cells)
    kernel = evaluate_minmod_slope_limiting_kernel(backend)
    kernel(ndrange=ncells)
    KernelAbstractions.synchronize(backend)
end
