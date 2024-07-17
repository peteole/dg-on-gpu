struct Euler <: Equation end
@declare_dofs Euler [:rho_u, :rho_v, :rho, :rho_E]

struct SodShockTube <: Scenario end

function evaluate_flux(eq::Euler, celldofs::SMatrix{basissize_nd,ndofs,F}, cellflux) where {basissize_nd, ndofs, F}
    s = EulerShortcuts()
    ρu = celldofs[:,s.rho_u]
    ρv = celldofs[:,s.rho_v]
    ρ = celldofs[:,s.rho]
    ρE = celldofs[:,s.rho_E]

    γ = 1.4
    p = (γ - 1)*(ρE - 1 ./ (2*ρ) .* (ρu.^2 .+ ρv.^2))


    for i in 1:basissize_nd
        # first update x flux
        cellflux[i,s.rho_u] = 1 / ρ[i] * ρu[i] * ρv[i] + p[i]
        cellflux[i,s.rho_v] = 1 / ρ[i] *ρu[i] * ρv[i]
        cellflux[i,s.rho] = ρu[i]
        cellflux[i,s.rho_E] = ρu[i] / ρ[i] * (ρE[i] + p[i])

        # then y flux
        cellflux[i + basissize_nd,s.rho_u] = 1 / ρ[i] * ρu[i] * ρv[i]
        cellflux[i + basissize_nd,s.rho_v] = 1 / ρ[i] * ρu[i] * ρv[i] + p[i]
        cellflux[i + basissize_nd,s.rho] = ρv[i]
        cellflux[i + basissize_nd,s.rho_E] = ρv[i] / ρ[i] * (ρE[i] + p[i])
    end
    

end

function max_eigenval(eq::Euler, celldata::SMatrix{order_sq,ndof,F}, normalidx::Int)::F where {order_sq,ndof,F}
    s = EulerShortcuts()
    ρu = SVector{order_sq,F}(@view celldata[:,s.rho_u])
    ρv = SVector{order_sq,F}(@view celldata[:,s.rho_v])
    ρ = SVector{order_sq,F}(@view celldata[:,s.rho])
    ρE = SVector{order_sq,F}(@view celldata[:,s.rho_E])

    γ = 1.4
    p = (γ - 1)*(ρE - 1 ./ (2*ρ) .* (ρu.^2 .+ ρv.^2))
    @assert all(p .>= 0)# "Pressure must be positive: p=$p, ρ=$ρ, ρu=$ρu, ρv=$ρv"

    v_n = (normalidx == 1 ? ρu ./ ρ : ρv ./ ρ)
    c = v_n .+ sqrt.(γ*p ./ celldata[:,s.rho])

    return maximum(c)
end

function evaluate_energy(eq::Euler, ρu, ρv, ρ, p)
    # assuming this is supposed to be the energy, not energy density?
    γ = 1.4
    p / (γ - 1) + 1 ./ (2*ρ) .* (ρu.^2 + ρv.^2)
end


# Gaussian Wave Scenario 
function is_periodic_boundary(eq::Euler, scenario::GaussianWave)
    true
end

function is_analytical_solution(equation::Euler, scenario::GaussianWave)
    false
end

function get_initial_values(eq::Euler, scenario::GaussianWave, global_position; t=0.0)
    pxg, pyg = global_position
    p = exp(-100 * (pxg - 0.5)^2 - 100 *(pyg - 0.5)^2) + 1
    rho = 1.0
    rhoE = evaluate_energy(eq, 0,0,rho,p)
    return @SVector[0.0, 0.0, 1.0, rhoE]
end

function evaluate_boundary(eq::Euler, scenario::GaussianWave, face, normalidx::Int, dofsface::SMatrix{num_2d_quadpoints,ndofs,F})::SMatrix{num_2d_quadpoints,ndofs,F} where {F,num_2d_quadpoints, ndofs}
    # dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
    # you need to set dofsfaceneigh

    dofsfaceneigh = SMatrix{num_2d_quadpoints,ndofs,F}(dof==normalidx ? -dofsface[qp,dof] : dofsface[qp,dof] for qp in 1:num_2d_quadpoints, dof in 1:ndofs)
    return dofsfaceneigh
end

# SodShockTube Scenario
function is_periodic_boundary(eq::Euler, scenario::SodShockTube)
    false
end

function is_analytical_solution(equation::Euler, scenario::SodShockTube)
    false
end

function evaluate_boundary(eq::Euler, scenario::SodShockTube, face, normalidx::Int, dofsface::SMatrix{num_2d_quadpoints,ndofs,F})::SMatrix{num_2d_quadpoints,ndofs,F} where {F, num_2d_quadpoints, ndofs}
    s = EulerShortcuts()
    flip_velocity_idx = normalidx == 1 ? s.rho_u : s.rho_v
    dofsfaceneigh = SMatrix{num_2d_quadpoints,ndofs,F}(dof==flip_velocity_idx ? -dofsface[qp,dof] : dofsface[qp,dof] for qp in 1:num_2d_quadpoints, dof in 1:ndofs)
    return dofsfaceneigh
end

function get_initial_values(eq::Euler, scenario::SodShockTube, global_position; t=0.0)
    pxg, pyg = global_position
    if pxg < 0.5 || pyg < 0.5
        rho = 0.125
        p = 0.1
    else
        rho = 1.0
        p = 1.0
    end
    rhoE = evaluate_energy(eq, 0,0,rho,p)
    return @SVector[0.0, 0.0, rho, rhoE]
end
