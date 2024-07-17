using LinearAlgebra
using StaticArrays
using KernelAbstractions

""" 
    project_to_face(projection_vector_0,projection_vector_1, dofs, faceval)

Projects the dofs to the face using the projection vectors and the bottom face.
Uses sum factorization to avoid unnecessary allocations.

The `Val` argument is used to dispatch on the face at compile time.
"""
function project_to_face(projection_vector_0::SVector{order,F}, projection_vector_1::SVector{order,F}, dofs::SMatrix{basis_size_nd,ndofs,F}, ::Val{TerraDG.bottom}) where {order,ndofs,basis_size_nd,F}
    li=LinearIndices((order,order))
    function compute_result(i,dof)
        result=F(0)
        for j2 in 1:order
            result+=projection_vector_0[j2]*dofs[li[i,j2],dof]
        end
        return result
    end
    return SMatrix{order,ndofs,F}(compute_result(i,dof) for i in 1:order, dof in 1:ndofs)
end


"""
    project_to_face(projection_vector_0,projection_vector_1, dofs, faceval)

Projects the dofs to the face using the projection vectors and the top face.
Uses sum factorization to avoid unnecessary allocations.

"""
function project_to_face(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, dofs::SMatrix{basis_size_nd,ndofs,F}, ::Val{TerraDG.top}) where {order,ndofs,basis_size_nd,F}
    li=LinearIndices((order,order))
    function compute_result(i,dof)
        result=F(0)
        for j2 in 1:order
            result+=projection_vector_1[j2]*dofs[li[i,j2],dof]
        end
        return result
    end
    return SMatrix{order,ndofs,F}(compute_result(i,dof) for i in 1:order, dof in 1:ndofs)
end

"""
    project_to_face(projection_vector_0,projection_vector_1, dofs, faceval)

Projects the dofs to the face using the projection vectors and the left face.
Uses sum factorization to avoid unnecessary allocations.

"""
function project_to_face(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, dofs::SMatrix{basis_size_nd,ndofs,F}, ::Val{TerraDG.left}) where {order,ndofs,basis_size_nd,F}
    li=LinearIndices((order,order))
    function compute_result(i,dof)
        result=F(0)
        for j2 in 1:order
            result+=projection_vector_0[j2]*dofs[li[j2,i],dof]
        end
        return result
    end
    return SMatrix{order,ndofs,F}(compute_result(i,dof) for i in 1:order, dof in 1:ndofs)
end

"""
    project_to_face(projection_vector_0,projection_vector_1, dofs, faceval)

Projects the dofs to the face using the projection vectors and the right face.
Uses sum factorization to avoid unnecessary allocations.

"""
function project_to_face(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, dofs::SMatrix{basis_size_nd,ndofs,F}, ::Val{TerraDG.right}) where {order,ndofs,basis_size_nd,F}
    li=LinearIndices((order,order))
    function compute_result(i,dof)
        result=F(0)
        for j2 in 1:order
            result+=projection_vector_1[j2]*dofs[li[j2,i],dof]
        end
        return result
    end
    return SMatrix{order,ndofs,F}(compute_result(i,dof) for i in 1:order, dof in 1:ndofs)
end


"""
    project_flux_to_face(face_projection_matrix_for_face, flux)

Projects the flux to the face using the sum factorization. First splits the flux into x and y components, then projects them to the face, and finally concatenates them.
"""
function project_flux_to_face(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F},
    flux::SMatrix{basis_size_nd2,ndofs,F}, faceval) where {order,ndofs,basis_size_nd2,F}
    basis_size_nd = order^2
    x_flux=SMatrix{basis_size_nd,ndofs,F}(@view flux[1:basis_size_nd, 1:ndofs])
    x_face_flux = project_to_face(projection_vector_0,projection_vector_1, x_flux, faceval)
    y_flux=SMatrix{basis_size_nd,ndofs,F}(@view flux[basis_size_nd+1:basis_size_nd2, 1:ndofs])
    y_face_flux = project_to_face(projection_vector_0,projection_vector_1, y_flux, faceval)
    concat::SMatrix{order*2,ndofs,F} = vcat(x_face_flux, y_face_flux)
    #@assert size(concat) == (basis_size_1d*2, ndofs)
    return concat
end

"""
    project_face_to_inner(projection_vector_0,projection_vector_1, facedofs, faceval)

Projects the face dofs to the inner dofs using the projection vectors. Legacy version, using sum factorization.

Same as `projection_matrix' * facedofs`
"""
function project_face_to_inner(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, facedofs::SMatrix{order,ndofs,F}, ::Val{TerraDG.bottom})where {order,ndofs,F}
    ci=CartesianIndices((order,order))
    return SMatrix{order^2,ndofs,F}(projection_vector_0[ci[i][2]]*facedofs[ci[i][1],dof] for i in 1:order^2, dof in 1:ndofs)
end

function project_face_to_inner(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, facedofs::SMatrix{order,ndofs,F}, ::Val{TerraDG.top}) where {order,ndofs,F}
    ci=CartesianIndices((order,order))
    return SMatrix{order^2,ndofs,F}(projection_vector_1[ci[i][2]]*facedofs[ci[i][1],dof] for i in 1:order^2, dof in 1:ndofs)
end

function project_face_to_inner(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, facedofs::SMatrix{order,ndofs,F}, ::Val{TerraDG.left}) where {order,ndofs,F}
    ci=CartesianIndices((order,order))
    return SMatrix{order^2,ndofs,F}(projection_vector_0[ci[i][1]]*facedofs[ci[i][2],dof] for i in 1:order^2, dof in 1:ndofs)
end

function project_face_to_inner(projection_vector_0::SVector{order,F},projection_vector_1::SVector{order,F}, facedofs::SMatrix{order,ndofs,F}, ::Val{TerraDG.right}) where {order,ndofs,F}
    ci=CartesianIndices((order,order))
    return SMatrix{order^2,ndofs,F}(projection_vector_1[ci[i][1]]*facedofs[ci[i][2],dof] for i in 1:order^2, dof in 1:ndofs)
end




"""
This version of the face integral relies on a certain order of the cells, as created in grid.jl.
We iterate over all faces and compute the numerical fluxes and the updates for the cells neighboring the face.
This function uses sum factorizations.

For a more simple and compatible version, see the `cuda_derivative_factorization` branch.
"""
function evaluate_face_integrals(eq::Equation, scenario::Scenario, globals::GlobalMatrices, grid::TerraDG.Grid{F,T1,T2,2,ndofs,order,T3}, dofsupdate::T2, dofs::T2) where {F<:Real,T1<:AbstractArray,T2<:AbstractArray,ndofs,order,T3}
    quadweights_1d = globals.quadweights_1d
    inv_massmatrix = globals.inv_reference_massmatrix_cell

    eigenvals = grid.eigenvals
    flux = grid.flux
    cells = grid.cells
    ncells = length(cells)
    gridsize_1d = grid.num_cells_1d
    neighbour_face_flux_buffer = grid.neighbour_face_flux_buffer
    face_integral_update_buffer = grid.face_integral_update_buffer


    face_val_left = Val(left)
    face_val_right = Val(right)
    face_val_bottom = Val(bottom)
    face_val_top = Val(top)

    # define all kernels
    @kernel function face_integral_inner(@Const(projection_vector_0), @Const(projection_vector_1), ::Val{normalidx}, ::Val{is_periodic}) where {normalidx, is_periodic}
        face_idx = @index(Global)

        # find the indices of the neighboring cells
        if is_periodic
            if normalidx == 1 #vertical
                cell_idx_1 = face_idx #left
                cell_idx_2 = div(face_idx - 1, gridsize_1d)*gridsize_1d + (face_idx % gridsize_1d + 1) % (gridsize_1d + 1) #right
                face_1, face_1_idx = face_val_right, 3
                face_2, face_2_idx = face_val_left, 1
            else #horizontal
                cell_idx_1 = face_idx #bottom
                cell_idx_2 = (face_idx + gridsize_1d - 1) % gridsize_1d^2 + 1 #top
                face_1, face_1_idx = face_val_top, 2
                face_2, face_2_idx = face_val_bottom, 4
            end
        else
            if normalidx == 1 #vertical
                cell_idx_1 = face_idx + div(face_idx - 1, gridsize_1d - 1) #left
                cell_idx_2 = cell_idx_1 + 1 #right
                face_1, face_1_idx = face_val_right, 3
                face_2, face_2_idx = face_val_left, 1

            else #horizontal
                cell_idx_1 =  face_idx #bottom
                cell_idx_2 =  face_idx + gridsize_1d #top
                face_1, face_1_idx = face_val_top, 2
                face_2, face_2_idx = face_val_bottom, 4
            end
        end

        # project dofs and flux to face
        dofs_face_1 = project_to_face(projection_vector_0, projection_vector_1, SMatrix{order^2,ndofs,F}(@view dofs[cell_idx_1, 1:order^2, 1:ndofs]), face_1)
        dofs_face_2 = project_to_face(projection_vector_0, projection_vector_1, SMatrix{order^2,ndofs,F}(@view dofs[cell_idx_2, 1:order^2, 1:ndofs]), face_2)
        flux_face_1 = project_flux_to_face(projection_vector_0, projection_vector_1, SMatrix{2*order^2,ndofs,F}(@view flux[cell_idx_1, 1:2*order^2, 1:ndofs]), face_1)
        flux_face_2 = project_flux_to_face(projection_vector_0, projection_vector_1, SMatrix{2*order^2,ndofs,F}(@view flux[cell_idx_2, 1:2*order^2, 1:ndofs]), face_2)

        # compute the maximum eigenvalue
        maxeigenval_1::F = max_eigenval(eq, dofs_face_1, normalidx) 
        maxeigenval_2::F = max_eigenval(eq, dofs_face_2, normalidx) 
        maxeigenval::F = max(maxeigenval_1, maxeigenval_2)

        eigenvals[cell_idx_1 + (face_1_idx - 1)*ncells] = maxeigenval
        eigenvals[cell_idx_2 + (face_2_idx - 1)*ncells] = maxeigenval

        # compute the numerical flux
        start_idx = (normalidx - 1)*order + 1 # = 1:order or order+1:2*order
        end_idx = normalidx * order

        avg_face_flux =  (SMatrix{order, ndofs, F}(@view flux_face_1[start_idx:end_idx, 1:ndofs]) .+ SMatrix{order, ndofs, F}(@view flux_face_2[start_idx:end_idx, 1:ndofs])) / F(2)
        diffusion = -1 / F(2) * maxeigenval .* (dofs_face_2 .- dofs_face_1)

        numericalflux = avg_face_flux + diffusion 

        # perform the update
        # 1st neighbor
        scaled_numerical_flux_1 = area(cells[cell_idx_1]) * quadweights_1d .* numericalflux
        scaled_numerical_flux_inner_1 = project_face_to_inner(projection_vector_0, projection_vector_1, scaled_numerical_flux_1, face_1)
        update_1 = 1 / volume(cells[cell_idx_1]) * inv_massmatrix * scaled_numerical_flux_inner_1
        for i in 1:order^2 
            for dof in 1:ndofs
                face_integral_update_buffer[cell_idx_1 + (face_1_idx - 1)*ncells, i, dof] = -1 * update_1[i, dof] 
            end
        end
        # 2nd neighbor
        scaled_numerical_flux_2 = area(cells[cell_idx_1]) * quadweights_1d .* numericalflux
        scaled_numerical_flux_inner_2 = project_face_to_inner(projection_vector_0, projection_vector_1, scaled_numerical_flux_2, face_2)
        update_2 = 1 / volume(cells[cell_idx_2]) * inv_massmatrix * scaled_numerical_flux_inner_2
        for i in 1:order^2 
            for dof in 1:ndofs
                face_integral_update_buffer[cell_idx_2 + (face_2_idx - 1)*ncells, i, dof] = update_2[i, dof] 
            end
        end
    end

    @kernel function face_integral_boundary(@Const(eq), @Const(scenario), @Const(projection_vector_0), @Const(projection_vector_1), ::Val{boundary}, ::Val{normalidx}) where {boundary, normalidx}
        face_idx = @index(Global)

        # find the index of the neighboring cell
        if boundary == 1 # left
            cell_idx = (face_idx - 1)*gridsize_1d + 1
            boundary_val = face_val_left
        elseif boundary == 2 # top
            cell_idx = face_idx + gridsize_1d * (gridsize_1d - 1)
            boundary_val = face_val_top
        elseif boundary == 3 # right
            cell_idx = face_idx * gridsize_1d
            boundary_val = face_val_right
        else # bottom
            cell_idx = face_idx
            boundary_val = face_val_bottom
        end

        # project dofs and flux to face
        dofs_face = project_to_face(projection_vector_0, projection_vector_1, SMatrix{order^2,ndofs,F}(@view dofs[cell_idx, 1:order^2, 1:ndofs]), boundary_val)
        flux_face = project_flux_to_face(projection_vector_0, projection_vector_1, SMatrix{2*order^2,ndofs,F}(@view flux[cell_idx, 1:2*order^2, 1:ndofs]), boundary_val)

        # evaluate the boundary condition
        dofs_face_boundary=@inline evaluate_boundary(eq, scenario, boundary, normalidx, dofs_face)
        evaluate_flux(eq, dofs_face_boundary, @view(neighbour_face_flux_buffer[cell_idx + (boundary - 1) * ncells, 1:2*order, 1:ndofs]))
        flux_face_boundary=SMatrix{2*order,ndofs,F}(@view(neighbour_face_flux_buffer[cell_idx + (boundary - 1) * ncells, 1:2*order, 1:ndofs]))

        # compute the maximum eigenvalue
        maxeigenval_1::F = max_eigenval(eq, dofs_face, normalidx)
        maxeigenval_2::F = max_eigenval(eq, dofs_face_boundary, normalidx)
        maxeigenval::F = max(maxeigenval_1, maxeigenval_2)

        eigenvals[cell_idx + (boundary - 1)*ncells] = maxeigenval

        # compute the numerical flux
        start_idx = (normalidx - 1)*order + 1 # = 1:order or order+1:2*order
        end_idx = normalidx * order

        avg_face_flux =  (SMatrix{order, ndofs, F}(@view flux_face[start_idx:end_idx, 1:ndofs]) .+ SMatrix{order, ndofs, F}(@view flux_face_boundary[start_idx:end_idx, 1:ndofs])) / F(2)
        factor = (boundary == 2 || boundary == 3)* F(-1.) + (boundary == 1 || boundary == 4) * F(1.) # cange factor to 1 for left/bottom boundary
        diffusion = factor / F(2) * maxeigenval .* (dofs_face_boundary .- dofs_face)

        numericalflux = avg_face_flux + diffusion

        # perform the update
        scaled_numerical_flux = area(cells[cell_idx]) * quadweights_1d .* numericalflux
        scaled_numerical_flux_inner = project_face_to_inner(projection_vector_0, projection_vector_1, scaled_numerical_flux, boundary_val)
        update = factor / volume(cells[cell_idx]) * inv_massmatrix * scaled_numerical_flux_inner

        # face_integral_update_buffer[cell_idx + (boundary-1)*ncells, 1:order^2, 1:ndofs] .= update
        # We avoid broadcasting due to compatibility issues with ceratin GPUs
        for i in 1:order^2 
            for dof in 1:ndofs
                face_integral_update_buffer[cell_idx + (boundary-1)*ncells, i, dof] = update[i, dof] 
            end
        end


    end

    backend = get_backend(dofsupdate)
    n_inner_faces = Int((gridsize_1d - 1)*gridsize_1d)

    # launch the kernels
    if is_periodic_boundary(eq, scenario)
        n_faces = n_inner_faces + gridsize_1d
        kernel = face_integral_inner(backend)
        # vertical faces
        kernel(globals.projection_vector_0, globals.projection_vector_1, Val(1), Val(true), ndrange=n_faces)
        # horizontal faces
        kernel(globals.projection_vector_0, globals.projection_vector_1, Val(2), Val(true), ndrange=n_faces)
    else
        # inner cells
        kernel = face_integral_inner(backend)
        #   vertical faces
        kernel(globals.projection_vector_0, globals.projection_vector_1, Val(1), Val(false), ndrange=n_inner_faces)
        #   horizontal faces
        kernel(globals.projection_vector_0, globals.projection_vector_1, Val(2), Val(false), ndrange=n_inner_faces)

        # boundaries
        kernel = face_integral_boundary(backend)
        #   left
        kernel(eq, scenario, globals.projection_vector_0, globals.projection_vector_1, Val(Int(left)), Val(1), ndrange=gridsize_1d)
        #   top
        kernel(eq, scenario, globals.projection_vector_0, globals.projection_vector_1, Val(Int(top)), Val(2), ndrange=gridsize_1d)
        #   right
        kernel(eq, scenario, globals.projection_vector_0, globals.projection_vector_1, Val(Int(right)), Val(1), ndrange=gridsize_1d)
        #   bottom
        kernel(eq, scenario, globals.projection_vector_0, globals.projection_vector_1, Val(Int(bottom)), Val(2), ndrange=gridsize_1d)
    end

    # wait for results
    KernelAbstractions.synchronize(backend)

    #aggregate results
    grid.maxeigenval = maximum(eigenvals)
    @views dofsupdate .+= face_integral_update_buffer[1:ncells, 1:order^2, 1:ndofs] .+ face_integral_update_buffer[ncells+1:2*ncells, 1:order^2, 1:ndofs] .+ face_integral_update_buffer[2*ncells+1:3*ncells, 1:order^2, 1:ndofs] .+ face_integral_update_buffer[3*ncells+1:4*ncells, 1:order^2, 1:ndofs]
end
