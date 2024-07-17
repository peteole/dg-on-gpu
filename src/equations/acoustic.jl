using StaticArrays
struct Acoustic <: Equation end
@declare_dofs Acoustic [:u, :v, :pressure, :rho, :K]

struct GaussianWave <: Scenario end
struct ConstantWave <: Scenario end

"""
is_periodic_boundary(eq::Acoustic, scenario::GaussianWave)
The GaussianWave scenario does not require periodic boundary conditions.
"""
function is_periodic_boundary(eq::Acoustic, scenario::GaussianWave)
    false
end

function evaluate_boundary(eq::Acoustic, scenario::GaussianWave, face, normalidx::Int, dofsface::SMatrix{num_2d_quadpoints,ndofs,F})::SMatrix{num_2d_quadpoints,ndofs,F} where {F,num_2d_quadpoints, ndofs}
    # dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
    # you need to set dofsfaceneigh

    dofsfaceneigh = SMatrix{num_2d_quadpoints,ndofs,F}(dof==normalidx ? -dofsface[qp,dof] : dofsface[qp,dof] for qp in 1:num_2d_quadpoints, dof in 1:ndofs)
    return dofsfaceneigh
end

function get_initial_values(eq::Acoustic, scenario::GaussianWave, global_position; t=0.0)
    x,y=global_position
    # s = AcousticShortcuts()
    
    # initial_values=zeros(get_ndofs(eq))
    # initial_values[s.u]=0
    # initial_values[s.v]=0
    # initial_values[s.pressure]=exp(-100*(x-0.5)^2-100*(y-0.5)^2)
    # initial_values[s.rho]=1
    # initial_values[s.K]= x<= 0.5 ? 1/5 : 1
    # return initial_values
    #return @SVector [0.0, 0.0, 1.0,2.0,3.0]
    return @SVector [0.0, 0.0, exp(-100.0*(x-0.5)^2-100*(y-0.5)^2), 1.0, x<= 0.5 ? 1.0/5.0 : 1.0]
end

function is_analytical_solution(equation::Acoustic, scenario::GaussianWave)
    false
end
function evaluate_boundary(eq::Acoustic, scenario::ConstantWave, face, normalidx::Int, dofsface::SMatrix{num_2d_quadpoints,ndofs,F})::SMatrix{num_2d_quadpoints,ndofs,F} where {F,num_2d_quadpoints,ndofs}
    # dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
    # you need to set dofsfaceneigh

    dofsfaceneigh = SMatrix{num_2d_quadpoints,ndofs,F}(dof==normalidx ? -dofsface[qp,dof] : dofsface[qp,dof] for qp in 1:num_2d_quadpoints, dof in 1:ndofs)
    return dofsfaceneigh
end

function get_initial_values(eq::Acoustic, scenario::ConstantWave, global_position; t=0.0)
    # s = AcousticShortcuts()
    
    # initial_values=zeros(get_ndofs(eq))
    # initial_values[s.u]=0
    # initial_values[s.v]=0
    # initial_values[s.pressure]=1.0
    # initial_values[s.rho]=1
    # initial_values[s.K]= 1.0
    # return initial_values
    return @SVector [0.0, 0.0, 1.0, 1.0, 1.0]
end

function is_analytical_solution(equation::Acoustic, scenario::ConstantWave)
    true
end


function evaluate_flux(eq::Acoustic, celldofs::SMatrix{basissize_nd,ndofs,F}, cellflux) where {basissize_nd, ndofs, F}
    s = AcousticShortcuts()
    l=basissize_nd
    v_x = celldofs[:,s.u]
    v_y = celldofs[:,s.v]
    p = celldofs[:,s.pressure]
    ρ0 = celldofs[:,s.rho]
    K0 = celldofs[:,s.K]

    # first update x flux
    cellflux[1:l,s.u] .= p ./ ρ0
    cellflux[1:l,s.v] .= 0
    cellflux[1:l,s.pressure] .= K0 .* v_x
    cellflux[1:l,s.rho] .= 0
    cellflux[1:l,s.K] .= 0
    
    # then y flux
    cellflux[l+1:end,s.u] .= 0
    cellflux[l+1:end,s.v] .= p ./ ρ0
    cellflux[l+1:end,s.pressure] .= K0 .* v_y
    cellflux[l+1:end,s.rho] .= 0
    cellflux[l+1:end,s.K] .= 0

    #cellflux .= [[p/ρ0, 0, K0*v_x, 0, 0], [0, p/ρ0, K0*v_y, 0, 0]
end

function max_eigenval(eq::Acoustic, celldata, normalidx)
    s = AcousticShortcuts()
    cs=sqrt(celldata[1,s.K] / celldata[1,s.rho])
    return cs
end