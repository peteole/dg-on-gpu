struct Advection <: Equation end
@declare_dofs Advection [:ρ1, :ρ2, :ρ3]

struct PlanarWaves <: Scenario
end

function is_periodic_boundary(equation::Advection, scenario::PlanarWaves)
    true
end

function get_initial_values(eq::Advection, scenario::PlanarWaves, global_position::NTuple{ndims,F}; t::F=F(0)) where {ndims,F}
    x, y= global_position .- t
    @SVector [sin(F(2) * π * (x + y)), sin(F(2) * π * y), F(1)]
end

function is_analytical_solution(equation::Advection, scenario::PlanarWaves)
    true
end

function evaluate_boundary(eq::Advection, scenario, face, normalidx::Int, dofsface::SMatrix{num_2d_quadpoints,ndofs,F})::SMatrix{num_2d_quadpoints,ndofs,F} where {F,num_2d_quadpoints, ndofs}
    # dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
    # you need to set dofsfaceneigh

    dofsfaceneigh = SMatrix{num_2d_quadpoints,ndofs,F}(dof==normalidx ? -dofsface[qp,dof] : dofsface[qp,dof] for qp in 1:num_2d_quadpoints, dof in 1:ndofs)
    
    return dofsfaceneigh
end


function evaluate_flux(eq::TerraDG.Advection, celldofs::SMatrix{basis_size_nd,ndofs,F}, cellflux) where {F,basis_size_nd,ndofs}
    s = TerraDG.AdvectionShortcuts()
    ρ1 = celldofs[:, s.ρ1]
    ρ2 = celldofs[:, s.ρ2]
    ρ3 = celldofs[:, s.ρ3]
    a = 1
    dim1_indices = 1:basis_size_nd
    dim2_indices = (basis_size_nd+1):2*basis_size_nd
    cellflux[dim1_indices, s.ρ1] .= a * ρ1
    cellflux[dim1_indices, s.ρ2] .= a * ρ2
    cellflux[dim1_indices, s.ρ3] .= a * ρ3
    cellflux[dim2_indices, s.ρ1] .= a * ρ1
    cellflux[dim2_indices, s.ρ2] .= a * ρ2
    cellflux[dim2_indices, s.ρ3] .= a * ρ3
end

function max_eigenval(eq::Advection, celldata, normalidx)
    # Is actually correct!
    1
end