@testset "Acoustic Equation Constant Wave" begin
    configfile = "./configs/constantwave.yaml"
    config = TerraDG.Configuration(configfile)

    eq = TerraDG.make_equation(config)
    scenario = TerraDG.make_scenario(config)

    @testset "Is Periodic Boundary" begin
        is_periodic = TerraDG.is_periodic_boundary(eq, scenario)
        @test is_periodic == false
    end

    @testset "Initial Values" begin
        global_position = (0.5, 0.5)
        result = TerraDG.get_initial_values(eq, scenario, global_position; t=0.0)
        @test result == @SVector [0.0, 0.0, 1.0, 1.0, 1.0]
    end

    @testset "Is Analitical Solution" begin
        result = TerraDG.is_analytical_solution(eq, scenario)
        @test result == true
    end

    @testset "Evaluate Boundaries" begin
        num_2d_quadpoints = 4
        ndofs = 3
        normalidx = 1
        dofsface = SMatrix{num_2d_quadpoints, ndofs, Float64}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
        result = TerraDG.evaluate_boundary(eq, scenario, :some_face, normalidx, dofsface)
        expected_result = SMatrix{num_2d_quadpoints, ndofs, Float64}((-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
        @test result == expected_result
    end

    @testset "Evaluate Flux" begin
        basis_size_nd = 3
        ndofs = 5
        celldofs = SMatrix{basis_size_nd, ndofs, Float64}([
            2.0, 4.0, 6.0, 8.0, 10.0,
            1.0, 2.0, 6.0, 4.0, 10.0,
            3.0, 4.0, 6.0, 6.0, 10.0
        ])
        cellflux = zeros(MMatrix{2 * basis_size_nd, ndofs, Float64})
        TerraDG.evaluate_flux(eq, celldofs, cellflux)

        expected_cellflux = SMatrix{2 * basis_size_nd, ndofs, Float64}([
            0.2, 2.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.2, 2.0, 1.0,
            12.0, 24.0, 60.0, 48.0, 60.0, 10.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])

        @test cellflux ≈ expected_cellflux
    end

    @testset "Max Eigenvalue" begin
        celldata = SMatrix{1, 5, Float64}([1.0, 2.0, 3.0, 4.0, 5.0])
        normalidx = 1      
        result = TerraDG.max_eigenval(eq, celldata, normalidx)
        @test result ≈ sqrt(celldata[1, 5] / celldata[1, 4])
    end

end