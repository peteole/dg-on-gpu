using StaticArrays
using LinearAlgebra
using KernelAbstractions

function get_χ_matrix(cellsize, inverse_cell_jacobian, quadweights_1d::SVector{order,F}) where {F<:Real,order}
    cell_volume = prod(cellsize)
    basissize_1d = order
    basis_size_nd = order^2
    cat_indices = CartesianIndices((basissize_1d, basissize_1d))
    function χ_i(linear_index, jac_indix)
        x, y = Tuple(cat_indices[linear_index])
        w_x, w_y = quadweights_1d[x], quadweights_1d[y]
        return cell_volume * w_x * w_y * inverse_cell_jacobian[jac_indix, jac_indix]
    end
    χ = Diagonal(
        SVector{2 * basis_size_nd,F}(χ_i((li - 1) % basis_size_nd + 1, (li - 1) ÷ basis_size_nd + 1) for li in 1:(2*basis_size_nd))
    )
    return χ
end

function compute_x_derivative_factorized(derivative_1d::SMatrix{basissize_1d,basissize_1d,F},flux::SMatrix{basissize_nd,ndofs,F}) where {ndofs,basissize_1d,basissize_nd,F}
    ci=CartesianIndices((basissize_1d,basissize_1d))
    li=LinearIndices((basissize_1d,basissize_1d))
    function compute_result(j,dof)
        j1,j2=Tuple(ci[j])
        result=F(0)
        for p1 in 1:basissize_1d
            result+=derivative_1d[j1,p1]*flux[li[p1,j2],dof]
        end
        return result
    end
    return SMatrix{basissize_nd,ndofs,F}(compute_result(j,dof) for j in 1:basissize_nd, dof in 1:ndofs)
end

function compute_y_derivative_factorized(derivative_1d::SMatrix{basissize_1d,basissize_1d,F},flux::SMatrix{basissize_nd,ndofs,F})where {ndofs,basissize_1d,basissize_nd,F}
    ci=CartesianIndices((basissize_1d,basissize_1d))
    li=LinearIndices((basissize_1d,basissize_1d))
    function compute_result(j,dof)
        j1,j2=Tuple(ci[j])
        result=F(0)
        for p2 in 1:basissize_1d
            result+=derivative_1d[j2,p2]*flux[li[j1,p2],dof]
        end
        return result
    end
    return SMatrix{basissize_nd,ndofs,F}(compute_result(j,dof) for j in 1:basissize_nd, dof in 1:ndofs)
end

function compute_derivate_factorized(derivative_1d::SMatrix{basissize_1d,basissize_1d,F},flux::SMatrix{basissize_nd2,ndofs,F}) where {ndofs,basissize_1d,basissize_nd2,F}
    basissize_nd = basissize_1d^2
    x_flux=SMatrix{basissize_nd,ndofs,F}(@view flux[1:basissize_nd,1:ndofs])
    y_flux=SMatrix{basissize_nd,ndofs,F}(@view flux[basissize_nd+1:basissize_nd2,1:ndofs])
    x_derivate_factorized=compute_x_derivative_factorized(derivative_1d,x_flux)
    y_derivate_factorized=compute_y_derivative_factorized(derivative_1d,y_flux)
    return x_derivate_factorized .+ y_derivate_factorized
end
"""
    evaluate_volumes(globals::GlobalMatrices, grid::Grid{F,T1,T2,2,ndofs,order,T3}, du::T2) where {F,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}

Evaluates the volume integrals for the DG-scheme on the grid `grid` with global matrices `globals` and update `du`.

Updates `du` in place.

Potentially runs on the GPU through a kernel.
"""
function evaluate_volumes(globals::GlobalMatrices, grid::Grid{F,T1,T2,2,ndofs,order,T3}, du::T2) where {F,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    cells = grid.cells
    flux = grid.flux
    derivative_1d = globals.derivative_1d
    inv_reference_massmatrix = globals.inv_reference_massmatrix_cell
    backend = get_backend(du)
    χ = globals.χ_matrix

    @kernel function evaluate_cell_volume(du, @Const(flux))
        cell_id = @index(Global)

        cellflux = SMatrix{2 * order^2,ndofs,F}(@view flux[cell_id, 1:2*order^2, 1:ndofs])
        cell::Cell{F,2,4} = cells[cell_id]
        scaled_fluxcoeff = χ * cellflux
        inv_elem_mass_matrix = 1/volume(cell) * inv_reference_massmatrix
        flux_derivative=compute_derivate_factorized(derivative_1d,scaled_fluxcoeff)
        change = SMatrix{order^2,ndofs,F}(inv_elem_mass_matrix * flux_derivative)
        old_du = SMatrix{order^2,ndofs,F}(@view du[cell_id, 1:order^2, 1:ndofs])
        for dof in 1:ndofs
            for p in 1:order^2
                du[cell_id, p, dof] += change[p,dof]
            end
        end
    end
    kernel = evaluate_cell_volume(backend)
    kernel(du,flux,ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end


