using FastGaussQuadrature
using StaticArrays
using LinearAlgebra

"""
    lagrange_1d(points, i, x)

Evaluate the Lagrange interpolation polynomial defined over nodal `points` with index `i`
at point `x`.
"""
function lagrange_1d(points, i, x)
    prod(x-p for (j,p) in enumerate(points) if j!=i;init=1)/prod(points[i]-p for (j,p) in enumerate(points) if j!=i;init=1)
end

function lagrange_1d_gpu(order, points, i, x)
    result = 1
    for j in 1:order
        p = points[j]
        if j!=i
            result *= (x-p)/(points[i] - p)
        end
    end
    result
end

"""
    lagrange_diff(points, i, x)

Evaluate the derivative of the Lagrange interpolation polynomial defined over nodal `points` 
with index `i` at point `x`.
"""
function lagrange_diff(points, i, x)
    return sum(1/(points[i]-points[j])*prod((x-points[m])/(points[i]-points[m]) for m in eachindex(points) if (m!=i)&&(m!=j);init=1.0) for j in eachindex(points) if i!=j;init=0.0)
end

"""
    get_quadpoints(n)

Compute quadrature points and weights for Gaussian quadrature
of order `n`.
The points (and thus the corresponding weights) are normalized to the range
``[0.0, 1.0]``.

Return a tuple of `points, weights`.
"""
function get_quadpoints(n)
    x, w = gausslegendre(n)
    # They are scaled in [-1.0, 1.0]
    # and we need [0.0, 1.0]
    Tuple((x .+ 1)./2), Tuple(w ./ 2)
end

"""
    Basis

A standard 1-dimensional basis of `order::Integer`
with `quadpoints::Array{Float64,1}` and 
corresponding `quadweights::Array{Float64,1}`
Is the basis (pun intended) for tensor-product
bases.

    Basis(order::Integer, dimensions)

Initialize a basis of `order::Integer` and
dimensions `dimensions`.
"""
struct Basis{F<:Real,order}
    quadpoints::NTuple{order,F}
    quadweights::NTuple{order,F}
    order::Int64
    dimensions::Int64

    function Basis(order, dimensions, ::Type{F}=Float32) where {F<:Real}
        quadpoints, quadweights = get_quadpoints(order)
        new{F,order}(F.(quadpoints), F.(quadweights), order, dimensions)
    end
end


"""
    Base.length(basis::Basis)

Return number of points for `basis` in n-dimensions.
"""
Base.length(basis::Basis) = prod(size(basis))

"""
    Base.size(basis::Basis)

Return number of points for `basis` for each dimensions as tuple.
"""
function Base.size(basis::Basis{F,order}) where {F,order}
    (order,order)
end

"""
    Base.size(basis::Basis, dim)

Return number of points for `basis` for dimensions `dim`.
"""
function Base.size(basis::Basis{F,order}, dim::Integer) where {F,order}
    order
end

"""
    evaluate_basis(basis::Basis, coeffs, x)

Evaluate the `basis` with coefficients
`coeffs` at point `x`.
"""
function evaluate_basis(basis::Basis{F,order}, coeffs, x) where {F,order}
    res=F(0)
    for (linear_index, cartesian_index) in enumerate(CartesianIndices(size(basis)))
        res += coeffs[linear_index] * (lagrange_1d_gpu(order,basis.quadpoints,cartesian_index[1],x[1]) * lagrange_1d_gpu(order,basis.quadpoints,cartesian_index[2],x[2]))
    end
    return res
end


"""
    project_to_reference_basis(fun, basis::Basis, ndofs)

Project the result of the function `fun` to coefficients
of the basis built of a tensor-product of `basis`.
The function `fun(x,y)`  takes in the ``x, y``-coordinates
and returns a vector with size `ndofs`.
The corresponding coefficients are returned.
"""
function project_to_reference_basis(fun, basis::Basis, ndofs)
    # only works for 2D but I think that is intended in the worksheet
    @assert basis.dimensions==2 "Only implemented for 2d basis but found $(basis.dimensions)d"
    basissize_1d = size(basis, 1)
    coeffs = zeros(basissize_1d^2, ndofs)
    lin_idx = LinearIndices((basissize_1d, basissize_1d))
    for(i,j) in Tuple.(CartesianIndices((basissize_1d, basissize_1d)))
        coeffs[lin_idx[i, j], :] .= fun(basis.quadpoints[i], basis.quadpoints[j])
    end
    return coeffs
end


"""
    massmatrix(basis, dimensions)

Return the mass-matrix for a `dimensions`-dimensional
tensor-product basis built up from the 1d-basis `basis`.
"""
function massmatrix(basis::Basis{F,order}, dimensions) where {F,order}
    basissize_1d = order
    #@assert dimensions==2 "Only implemented for 2d"
    diag_entries = SVector{basissize_1d^2,F}(basis.quadweights[pos[1]] * basis.quadweights[pos[2]] for pos in CartesianIndices((basissize_1d, basissize_1d)))
    return Diagonal(diag_entries)
end

"""
    derivativematrix(basis)

Returns the 2-dimensional derivative matrix for `basis`.
Multiplying this with flux-coefficients of shape
`(dimensions * basissize_2d, ndofs)` returns the
coefficients of the corresponding derivative.
"""
function derivativematrix(basis)
    basissize_1d = size(basis, 1)
    points_1D,_ = get_quadpoints(basissize_1d)
    Dx = zeros(length(basis), length(basis))
    Dy = zeros(length(basis), length(basis))
    lin_idx = LinearIndices((basissize_1d, basissize_1d))
    
    for pos in CartesianIndices((basissize_1d, basissize_1d))
        for basis in CartesianIndices((basissize_1d, basissize_1d))
            x_index,y_index = Tuple(pos)
            x,y=points_1D[x_index],points_1D[y_index]
            i,j = Tuple(basis)
            Dx[lin_idx[basis], lin_idx[pos]] = lagrange_diff(points_1D,i, x) * lagrange_1d(points_1D, j, y)
            Dy[lin_idx[basis], lin_idx[pos]] = lagrange_1d(points_1D,i, x) * lagrange_diff(points_1D, j, y)
        end
    end
    hcat(Dx,Dy)
end

#@enum Face left=1 top=2 right=3 bottom=4
"""
    get_face_quadpoints(basis::Basis, face)

Return the quadrature points at the face `face` for basis `basis`.
"""
function get_face_quadpoints(basis::Basis, face)
    @assert basis.dimensions==2 "Only implemented for 2d"
    if face==TerraDG.bottom
        return [(p,0.0) for p in basis.quadpoints]
    elseif face==top
        return [(p,1.0) for p in basis.quadpoints]
    elseif face==right
        return [(1.0,p) for p in basis.quadpoints]
    elseif face==left
        return [(0.0,p) for p in basis.quadpoints]
    end
end

"""
    face_projection_matrix(basis, face)

Return the face projection matrix for `basis` and `face`.
Multiplying it with coefficient vector for the right basis
returns the coefficients of the solution evaluated at the 
quadrature nodes of the face.
"""
function face_projection_matrix(basis::Basis, face)
    basissize_1d = size(basis, 1)
    cartesian_indices=CartesianIndices((basissize_1d, basissize_1d))
    face_points=get_face_quadpoints(basis,face)
    [lagrange_1d(basis.quadpoints,cartesian_indices[j][1],face_points[i][1])*lagrange_1d(basis.quadpoints,cartesian_indices[j][2],face_points[i][2]) for i in eachindex(face_points), j in 1:length(cartesian_indices)]
end

"""
    evaluate_m_to_n_vandermonde_basis(basis) 

Return the Vandermonde matrix that converts between the 2D-modal 
(normalized) Legendre-basis and the 2D-nodal tensor-product basis built
with `basis`.
"""
function evaluate_m_to_n_vandermonde_basis(basis)
    Array{Float64}(I, length(basis), length(basis))
end


function quadweights_nd(basis::Basis)
    quadweights=zeros(length(basis))
    @assert basis.dimensions==2 "Only implemented for 2d basis but found $(basis.dimensions)d"
    linear_indices = LinearIndices(size(basis))

    for ci in CartesianIndices(size(basis))
        quadweights[linear_indices[ci]] = basis.quadweights[ci[1]] * basis.quadweights[ci[2]]
    end
    quadweights
end