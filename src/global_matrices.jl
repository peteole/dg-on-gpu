using StaticArrays
"""
    struct GlobalMatrices

Stores global matrices and some other things.
Everything that is very expensive to compute which stays
constant throughout the simulation should be stored here.


    GlobalMatrices(basis::Basis, filter::Filter, dimensions)

Initialize GlobalMatrices for `basis`, `filter` and `dimensions`.
"""
struct GlobalMatrices{F<:Real,basis_size_1d,basis_size_1d2,basis_size_nd,basis_size_nd2}
    normalsigns::Dict{Face,Int64}
    normalidxs::Dict{Face,Int64}
    oppositefaces::Dict{Face,Face}

    quadweights_1d::SVector{basis_size_1d,F}
    inv_reference_massmatrix_cell::AbstractMatrix{F}
    # only works for equally sized cells
    χ_matrix::Diagonal{F,SVector{basis_size_nd2,F}}
    derivative_1d::SMatrix{basis_size_1d,basis_size_1d,F}
    
    projection_vector_0::SVector{basis_size_1d,F}

    projection_vector_1::SVector{basis_size_1d,F}

    function GlobalMatrices(basis::Basis{F,order}, grid::Grid{F,T1,T2,ndims,ndofs,order,T3}, dimensions=2) where {F<:Real,order,T1,T2,ndims,ndofs,T3}
        basis_size_1d = order
        basis_size_nd = order^dimensions
        basis_size_nd2 = 2 * basis_size_nd
        basis_size_1d2 = 2 * basis_size_1d

        normalsigns = Dict(left => -1, right => 1, top => 1, bottom => -1)
        normalidxs = Dict(left => 1, right => 1, top => 2, bottom => 2)
        oppositefaces = Dict(left => right, right => left, top => bottom, bottom => top)

        quadweights_1d = basis.quadweights

        reference_massmatrix_cell =massmatrix(basis, dimensions)
        inv_reference_massmatrix_cell = inv(reference_massmatrix_cell)

        χ_matrix = get_χ_matrix(grid.cellsize, inverse_jacobian(grid.cellsize), SVector{order,F}(quadweights_1d))
        points_1d=basis.quadpoints
        derivative_1d = SMatrix{basis_size_1d,basis_size_1d,F}(TerraDG.lagrange_diff(points_1d, i, points_1d[j]) for i in 1:basis_size_1d, j in 1:basis_size_1d)


        projection_vector_0 = SVector{basis_size_1d,F}(TerraDG.lagrange_1d(points_1d, i, 0) for i in 1:basis_size_1d)
        projection_vector_1 = SVector{basis_size_1d,F}(TerraDG.lagrange_1d(points_1d, i, 1) for i in 1:basis_size_1d)

        new{F,basis_size_1d,basis_size_1d2,basis_size_nd,basis_size_nd2}(normalsigns, normalidxs, oppositefaces, quadweights_1d,inv_reference_massmatrix_cell,
            χ_matrix, derivative_1d,
            projection_vector_0, projection_vector_1)
    end
end

