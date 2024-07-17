@testset "Sibson Equation" begin
    configfile = "./configs/interpolation.yaml"
    config = TerraDG.Configuration(configfile)

    eq = TerraDG.make_equation(config)
    scenario = TerraDG.make_scenario(config)

    @testset "Is Periodic Boundary" begin
        is_periodic = TerraDG.is_periodic_boundary(eq, scenario)
        @test is_periodic == true
    end

    @testset "Initial Values" begin
        global_position = (0.5, 0.5)
        result = TerraDG.get_initial_values(eq, scenario, global_position; t=0.0)
        @test result == @SVector [cos(4π * √((0.5 - 1 / 4)^2 + (0.5 - 1 / 4)^2))]
    end

    @testset "Is Analitical Solution" begin
        result = TerraDG.is_analytical_solution(eq, scenario)
        @test result == true
    end

    @testset "Evaluate Flux" begin
        basis_size_nd = 3
        ndofs = 3
        celldofs = SMatrix{basis_size_nd, ndofs, Float64}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        cellflux = zeros(MMatrix{2*basis_size_nd, ndofs, Float64})
        TerraDG.evaluate_flux(eq, celldofs, cellflux)
        @test cellflux == zeros(SMatrix{2*basis_size_nd, ndofs, Float64})
    end

    @testset "Max Eigenvalue" begin
        result = TerraDG.max_eigenval(eq, :some_celldata, 1)
        @test result == 0
    end

end