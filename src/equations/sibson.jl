struct Sibson <: Equation end
@declare_dofs Sibson [:f]

struct Interpolation <: Scenario
end

function is_periodic_boundary(equation::Sibson, scenario::Interpolation)
    true
end

function get_initial_values(eq::Sibson, scenario::Interpolation, global_position::NTuple{ndims,F}; t::F=F(0)) where {ndims,F}
    x, y = global_position .- (1/F(2), 1/F(2))
    @SVector [cos(4π*√((x - 1/F(4))^2 + (y - 1/F(4))^2))]
end

function is_analytical_solution(equation::Sibson, scenario::Interpolation)
    true
end

function evaluate_flux(eq::Sibson, celldofs, cellflux)
    0
end

function max_eigenval(eq::Sibson, celldata, normalidx)
    0
end
