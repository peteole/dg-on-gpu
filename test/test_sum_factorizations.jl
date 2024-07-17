using StaticArrays


function get_derivative_matrix(basis)
    basissize_1d = size(basis, 1)
    points_1D, _ = TerraDG.get_quadpoints(basissize_1d)
    Dx = zeros(length(basis), length(basis))
    Dy = zeros(length(basis), length(basis))
    lin_idx = LinearIndices((basissize_1d, basissize_1d))

    for pos in CartesianIndices((basissize_1d, basissize_1d))
        for basis in CartesianIndices((basissize_1d, basissize_1d))
            x_index, y_index = Tuple(pos)
            x, y = points_1D[x_index], points_1D[y_index]
            i, j = Tuple(basis)
            Dx[lin_idx[basis], lin_idx[pos]] = TerraDG.lagrange_diff(points_1D, i, x) * TerraDG.lagrange_1d(points_1D, j, y)
            Dy[lin_idx[basis], lin_idx[pos]] = TerraDG.lagrange_1d(points_1D, i, x) * TerraDG.lagrange_diff(points_1D, j, y)
        end
    end
    D = hcat(Dx, Dy)
    return D
end
@testset "Derivative sum factorizations are correct" begin
    config = TerraDG.Configuration("../src/input/advection.yaml")
    config.device = "cpu"
    for order in 1:5
        config.order = order
        eq, scenario, grid, slope_limiting_buffer, integrator, globals, F = TerraDG.setup_simulation(config)
        basis = grid.basis
        D=get_derivative_matrix(basis)
        derivative_1d = globals.derivative_1d
        testflux = @SMatrix rand(F, order^2*2, 3)
        full_derivative = D * testflux
        factorized_derivative = TerraDG.compute_derivate_factorized(derivative_1d, testflux)
        @test isapprox(factorized_derivative, full_derivative, rtol=1e-6)
    end
end

@testset "Face projection is correct" begin
    config = TerraDG.Configuration("../src/input/advection.yaml")
    config.device = "cpu"
    for order in 1:5
        config.order = order
        eq, scenario, grid, slope_limiting_buffer, integrator, globals, F = TerraDG.setup_simulation(config)
        basis = grid.basis
        projection_vector_0 = globals.projection_vector_0
        projection_vector_1 = globals.projection_vector_1
        testdofs = @SMatrix rand(F, order^2, 3)
        for face in [TerraDG.bottom, TerraDG.top, TerraDG.left, TerraDG.right]
            factorized_projection = TerraDG.project_to_face(projection_vector_0, projection_vector_1, testdofs, Val(face))
            full_projection_matrix = TerraDG.face_projection_matrix(basis, face)
            full_projection = full_projection_matrix * testdofs

            @test isapprox(factorized_projection, full_projection, rtol=1e-6)
        end
    end
end

function test_face_to_inner_projection(face)
    testdofs = @SMatrix rand(basissize_1d, 3)
    factorized_projection = project_face_to_inner(projection_vector_0, projection_vector_1, testdofs, Val(face))
    full_projection_matrix = TerraDG.face_projection_matrix(basis, face)
    full_projection = full_projection_matrix' * testdofs

    @assert isapprox(factorized_projection, full_projection, rtol=1e-6) "Factorized inner projection does not match full inner projection: $factorized_projection vs $full_projection"
end

@testset "Face to inner projection is correct" begin
    config = TerraDG.Configuration("../src/input/advection.yaml")
    config.device = "cpu"
    for order in 1:5
        config.order = order
        eq, scenario, grid, slope_limiting_buffer, integrator, globals, F = TerraDG.setup_simulation(config)
        basis = grid.basis
        projection_vector_0 = globals.projection_vector_0
        projection_vector_1 = globals.projection_vector_1
        testdofs = @SMatrix rand(F, order, 3)
        for face in [TerraDG.bottom, TerraDG.top, TerraDG.left, TerraDG.right]
            factorized_projection = TerraDG.project_face_to_inner(projection_vector_0, projection_vector_1, testdofs, Val(face))
            full_projection_matrix = TerraDG.face_projection_matrix(basis, face)
            full_projection = full_projection_matrix' * testdofs

            @test isapprox(factorized_projection, full_projection, rtol=1e-6)
        end
    end
end