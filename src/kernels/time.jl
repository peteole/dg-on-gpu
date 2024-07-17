"""
    abstract type TimeIntegrator

Abstract type for ODE-Integrators.
They should store the update as internal state to 
avoid costly reallocations.

# Implementation
For new integrators, the method `step` needs to be 
overwritten.
This method should modify the dofs stored in the grid
in-place.
"""
abstract type TimeIntegrator end

"""
    ExplicitEuler(grid::Grid)

Return Euler time-integrator for `grid`.
First order accurate.
"""
struct ExplicitEuler{F<:Real,T<:AbstractArray{F,3}} <: TimeIntegrator
    dofsupdate::T

    function ExplicitEuler(grid::Grid{F,T1,T2,ndims,ndofs,order,T3}) where {F<:Real,T1<:AbstractVector{<:Cell},T2<:AbstractArray{F,3},ndims,ndofs,order,T3<:AbstractVector{F}}
        dofsupdate = similar(grid.dofs)
        new{F,T2}(dofsupdate)
    end
end

"""
    step(f, integrator::ExplicitEuler, grid, dt)

Performs an update with the explicit euler method
on `grid` and timestepsize `dt`.
"""
function step(f, integrator::ExplicitEuler{F,T}, grid, dt::F) where {F<:Real,T<:AbstractArray{F,3}}
    integrator.dofsupdate .= F(0)
    f(integrator.dofsupdate, grid.dofs, grid.time)
    grid.dofs .+= dt .* integrator.dofsupdate
    @info "Update has" norm(integrator.dofsupdate)
end

"""
    SSPRK2(grid::Grid)

Return SSPRK2 time-integrator for `grid`.
Two-stage strong-stability preserving Runge Kutta method.
Second order accurate.
"""
struct SSPRK2{F<:Real,T<:AbstractArray{F,3}} <: TimeIntegrator
    stage1::T
    stage2::T

    function SSPRK2(grid::Grid{F,T1,T2,ndims,ndofs,order,T3}) where {F<:Real,T1<:AbstractVector{<:Cell},T2<:AbstractArray{F,3},ndims,ndofs,order,T3<:AbstractVector{F}}
        stage1 = similar(grid.dofs)
        stage2 = similar(grid.dofs)
        new{F,T2}(stage1, stage2)

    end
end

"""
    step(f, integrator::SSPRK2, grid, dt)

Performs an update with the SSPRK2 method
on `grid` and timestepsize `dt`.
"""

function step(f, integrator::SSPRK2{F,T}, grid, dt::F) where {F<:Real,T<:AbstractArray{F,3}}
    integrator.stage1 .= 0
    integrator.stage2 .= 0

    f(integrator.stage1, grid.dofs, grid.time)
    integrator.stage1 .= grid.dofs .+ dt .* integrator.stage1

    f(integrator.stage2, integrator.stage1, grid.time + dt)

    grid.dofs .= 0.5f0 .* (grid.dofs .+ integrator.stage1 .+
                         dt .* integrator.stage2)
end

"""
    SSPRK3(grid::Grid)

Return SSPRK3 time-integrator for `grid`.
Three-stage strong-stability preserving Runge Kutta method.
Third order accurate.
"""
struct SSPRK3{F<:Real,T<:AbstractArray{F,3}} <: TimeIntegrator
    stage1::T
    stage2::T
    stage3::T

    function SSPRK3(grid::Grid{F,T1,T2,ndims,ndofs,order,T3}) where {F<:Real,T1<:AbstractVector{<:Cell},T2<:AbstractArray{F,3},ndims,ndofs,order,T3<:AbstractVector{F}}
        stage1 = similar(grid.dofs)
        stage2 = similar(grid.dofs)
        stage3 = similar(grid.dofs)
        new{F,T2}(stage1, stage2, stage3)

    end
end

"""
    step(f, integrator::SSPRK3, grid, dt)

Performs an update with the SSPRK3 method
on `grid` and timestepsize `dt`.
"""
function step(f, integrator::SSPRK3{F,T}, grid, dt::F) where {F<:Real,T<:AbstractArray{F,3}}
    integrator.stage1 .= 0
    integrator.stage2 .= 0
    integrator.stage3 .= 0

    f(integrator.stage1, grid.dofs, grid.time)
    integrator.stage1 .= grid.dofs .+ dt .* integrator.stage1

    f(integrator.stage2, integrator.stage1, grid.time + dt)
    integrator.stage2 .= F(0.75) .* grid.dofs .+ F(0.25) .* (
        integrator.stage1 + dt * integrator.stage2)

    f(integrator.stage3, integrator.stage2, grid.time + F(0.5) * dt)


    grid.dofs .= F(1/3) .* grid.dofs .+ F(2/3)* (
        integrator.stage2 .+ dt .* integrator.stage3)
end

"""
    make_timeintegrator(config::Configuration, grid::Grid)

Returns time integrator from `config` for `grid`.
"""
function make_timeintegrator(config::Configuration, grid::Grid)
    if config.timeintegrator_name == "Euler"
        return ExplicitEuler(grid)
    elseif config.timeintegrator_name == "SSPRK2"
        return SSPRK2(grid)
    elseif config.timeintegrator_name == "SSPRK3"
        return SSPRK3(grid)
    else
        error(string("Unknown timeintegrator name: ", config.timeintegrator_name))
    end
end