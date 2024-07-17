using StaticArrays
"""
    @enum FaceType regular=1 boundary=2

Regular faces are faces that have another cell as neighbor.
Boundary faces are faces on the boundary, i.e., where we need
to construct a solution at each timestep.
"""
@enum FaceType regular = 1 boundary = 2

"""
    struct Cell
    
`Cell` stores all information about grid cells, such as their
center `center`, their size `size`.
It also contains information about its `neighbors`.
In case it is neighbored by a boundary cell, `facetypes` 
indicates this and the neighbor is undefined.
Finally, it stores the `dataidx` which denotes at which
position the data for the cell is stored in the relevant
data arrays.

The Cell is parameterized by the floating point type `F`, the number of dimensions `ndims` and the number of neighbors `num_neighbors`, making it fixed size.
"""
struct Cell{F,ndims,num_neighbors}
    center::NTuple{ndims,F}
    size::NTuple{ndims,F}
    neighbor_indices::NTuple{num_neighbors,Int64}
    facetypes::NTuple{num_neighbors,FaceType}
    dataidx::Int64
end

"""
    mutable struct Grid

`Grid` stores information about the grid and also information
about the whole simulation.
It contains the `basis` for the grid, the `cells` of the grid,
the `size` of the grid, the degrees of freedom `dofs` and the
fluxes `flux`.
It also contains the `maxeigenval` which is the maximum eigenvalue
of the grid, the current `time` of the simulation, the `eigenvals`
of the grid and the `neighbour_face_flux_buffer` which is used
for the surface integral.
Finally, it stores the `cellsize` of the cells.

It is parameterized by the floating point type `F`, the type of the cells `T1`, the type of the degrees of freedom `T2`, the number of dimensions `ndims`, the number of degrees of freedom `ndofs`, the order of the basis `order` and the type of the eigenvalues `T3`.
`T1`, `T2` and `T3` are arrays potentially stored on the GPU.
"""
mutable struct Grid{F,T1<:AbstractVector{<:Cell},T2<:AbstractArray{F,3},ndims,ndofs,order,T3<:AbstractVector{F}}
    basis::Basis{F,order}
    cells::T1
    size::NTuple{ndims,F}
    num_cells_1d::Int64
    dofs::T2
    flux::T2
    maxeigenval::F
    time::F
    eigenvals::T3
    neighbour_face_flux_buffer::T2
    face_integral_update_buffer::T2

    # we have fixed size cells
    cellsize::NTuple{ndims,F}
end

"""
Move all the data of the grid to the CPU, returns a new grid.
"""
function cpu(grid::Grid{F,T1,T2,ndims,ndofs,order,T3}) where {F<:Real,T1<:AbstractVector{<:Cell},T2<:AbstractArray{F,3},ndims,ndofs,order,T3<:AbstractVector{F}}
    return Grid{F,Vector{Cell{F,2,4}},Array{F,3},ndims,ndofs,order,T3}(grid.basis, Array(grid.cells), grid.size, grid.num_cells_1d, Array(grid.dofs), Array(grid.flux), grid.maxeigenval, grid.time, Vector(grid.eigenvals),Array(grid.neighbour_face_flux_buffer),Array(grid.face_integral_update_buffer), grid.cellsize)
end

"""
    @enum Face left=1 top=2 right=3 bottom=4

`Face` describes the ordering of our faces. The order is irrelevant
as long as the same order is used everywhere.
Using the wrong order leads to very hard bugs!
"""
@enum Face left = 1 top = 2 right = 3 bottom = 4

"""
    get_neighbor(eq, scenario, index, maxindex, offset)

Returns the index of the neighboring cell.
Here, `index` is the 2d-index of the current cell,
`maxindex` is the maximum index,
and `offset` is the difference in index of the neighbor.
As minimum index (0,0) is assumed.
Returns both the index of the neighbor and the type of the face.
It handles both periodic boundary cells (returns correct neighbor and regular
face type) and proper boundaries (returns periodic neighbor and boundary face type).

# Implementation
The type of boundaries depends on the `equation` and the `scenario`.
Users can overwrite the function is_periodic_boundary.
"""
function get_neighbor(eq::Equation, scenario::Scenario, index, maxindex, offset)
    facetype = regular
    i1 = index[1] + offset[1]
    i2 = index[2] + offset[2]
    if i1 < 1
        i1 = maxindex[1]
        facetype = boundary
    elseif i1 > maxindex[1]
        i1 = 1
        facetype = boundary
    end
    if i2 < 1
        i2 = maxindex[2]
        facetype = boundary
    elseif i2 > maxindex[2]
        i2 = 1
        facetype = boundary
    end
    if (is_periodic_boundary(eq, scenario))
        facetype = regular
    end
    CartesianIndex(i1, i2), facetype
end

"""
    make_mesh(eq, scenario, gridsize_1d, cellsize, offset)

Returns all cells of a mesh with number of cel(per dimension) given by 
`gridsize_1d`, size of each cell by `cellsize` and `offset` of grid.
"""
function make_mesh(eq, scenario, gridsize_1d, cellsize, offset::NTuple{2,F}, ::Type{T})::T{Cell{F,2,4}} where {F,T<:AbstractArray}
    gridsize = gridsize_1d^2
    cells_array = Vector{Cell{F,2,4}}(undef, gridsize)
    linear_grid_index = LinearIndices((gridsize_1d, gridsize_1d))
    for i in CartesianIndices((gridsize_1d, gridsize_1d))
        center = offset .+ (i[1], i[2]) .* cellsize .- cellsize ./ 2
        index_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        neighbors = Vector{Int64}(undef, 4)
        facetypes = Vector{FaceType}(undef, 4)
        for j = 1:4
            index_offset = index_offsets[j]
            neighbor, facetype = get_neighbor(eq, scenario, i, (gridsize_1d, gridsize_1d), index_offset)
            neighbors[j] = linear_grid_index[neighbor]
            facetypes[j] = facetype
        end
        cells_array[linear_grid_index[i]] = Cell(center, cellsize, Tuple(neighbors), Tuple(facetypes), linear_grid_index[i])
    end
    T{Cell{F,2,4},1}(cells_array)
end

# TODO: should we update docs? (E.g. with type?)
"""
    make_grid(eq::Equation, scenario::Scenario, gridsize_1d, size, order, ::Type{T}=Array,::Type{F}=Float64)


Returns a grid for equation `eq`, scenario `scenario`, with cells of size `size`
    and number of cells per dimension equals to `gridsize_1d`. The type of array and the floating point precision can be specified.
"""
function make_grid(eq::Equation, scenario::Scenario, gridsize_1d, size, order, ::Type{T}=Array,::Type{F}=Float64) where {F,T<:AbstractArray}
    gridsize = gridsize_1d^2
    dofs = T{F,3}(undef, (gridsize, order^2, get_ndofs(eq)))
    flux = similar(dofs, gridsize, order^2 * 2, get_ndofs(eq))
    cellsize = F.(size ./ gridsize_1d)
    cells = make_mesh(eq, scenario, gridsize_1d, cellsize, F.((0.0, 0.0)), T)
    basis = Basis(order, 2,F)
    eigenvals = T{F,1}(undef, gridsize * 4)
    neighbour_face_flux_buffer =T{F,3}(undef, (length(cells)*4, order*2, get_ndofs(eq)))
    face_integral_update_buffer =T{F,3}(undef, (length(cells)*4, order^2, get_ndofs(eq)))
    Grid{F,T{Cell{F,2,4},1},T{F,3},2,get_ndofs(eq),order,T{F,1}}(basis, cells, size, gridsize_1d, dofs, flux, -1.0, 0.0,eigenvals,neighbour_face_flux_buffer, face_integral_update_buffer, cellsize)
end

function get_cell_flux(grid::Grid{F,T1,T2,ndims,ndofs,order,T3}, cell_id::Int) where {F<:Real,T1<:AbstractVector{<:Cell},T2<:AbstractArray{F,3},ndims,ndofs,order,T3<:AbstractVector{F}}
    return SMatrix{order^2 * 2,ndofs}(@view grid.flux[:, :, cell_id])
end

"""
    make_grid(config::Configuration, eq::Equation, scenario::Scenario)

Returns a grid for configuration `config`, equation `eq` and scenario `scenario`
"""
function make_grid(config::Configuration, eq::Equation, scenario::Scenario, ::Type{T},::Type{F}=Float64) where {F<:Real,T<:AbstractArray}
    make_grid(eq,
        scenario,
        config.grid_elements,
        config.physicalsize,
        config.order, T, F)
end


"""
    globalposition(cell:Cell, coordinate_reference)
    
Returns the global positon of reference coordinates `coordinate_reference`
for a cell with center `cellcenter` and size `cellsize`.
"""
function globalposition(cell::Cell, coordinate_reference)
    if minimum(coordinate_reference) < 0 || maximum(coordinate_reference) > 1
        throw(BoundsError())
    end
    cell.size .* coordinate_reference .+ cell.center .- 0.5f0 .* cell.size
end

"""
    localposition(cell::Cell, coordinate_global)

Opposite of `globalposition`.
Returns the reference (local) positon for global coordinates
`coordinate_global` for a cell with center `cellcenter` and size `cellsize`.
"""
# TODO: It is not used
function localposition(cell::Cell, coordinate_global)
    coordinate_reference = [1.0 / cell.size[i] * (coordinate_global[i] - cell.center[i] + 0.5 * cell.size[i]) for i in eachindex(cell.center)]
    if minimum(coordinate_reference) < 0.0 || maximum(coordinate_reference) > 1.0
        throw(BoundsError())
    end
    coordinate_reference
end

"""
    volume(cell::Cell)

Returns the volume of a quad `cell`.
In 2D, it returns the area.
"""
function volume(cell::Cell)
    #@assert all(y -> y == cell.size[1], cell.size)
    prod(cell.size)
end

"""
    area(cell::Cell)

Returns the area of a quad `cell`.
In 1D, it returns the side-length.
"""
function area(cell::Cell{F,2,4}) where F
    @assert all(y -> y == cell.size[1], cell.size)
    area = F(1)
    for i = 2:length(cell.size)
        area *= cell.size[i]
    end
    area
end


"""
    inverse_jacobian(cellsize::NTuple{ndims,F})

Returns the inverse jacobian of a cell with size `cellsize`.
"""
function inverse_jacobian(cellsize::NTuple{ndims,F}) where {ndims, F}
    Diagonal(SVector{ndims,F}(1 ./ cellsize))
end