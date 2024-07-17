@testset "Euler Gaussian Equation" begin
    configfile = "./configs/euler_gaussian_wave.yaml"
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
        @test result ≈ @SVector[0.0, 0.0, 1.0, 5.0]
    end

    @testset "Is Analitical Solution" begin
        result = TerraDG.is_analytical_solution(eq, scenario)
        @test result == false
    end

    @testset "Evaluate Flux" begin
        basis_size_nd = 3
        ndofs = 4
        celldofs = SMatrix{basis_size_nd, ndofs, Float64}([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ])
        cellflux = zeros(MMatrix{2*basis_size_nd, ndofs, Float64})
        TerraDG.evaluate_flux(eq, celldofs, cellflux)
        expected_cellflux = SMatrix{2 * basis_size_nd, ndofs, Float64}((4.085714285714285, 4.924999999999999, 5.799999999999999, 0.5714285714285714, 1.25, 2.0, 0.5714285714285714, 1.25, 2.0, 4.085714285714285, 4.924999999999999, 5.799999999999999, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.930612244897959, 3.6687499999999997, 5.266666666666666, 7.722448979591836, 9.171875, 10.533333333333331))
        @test cellflux ≈ expected_cellflux
    end

    @testset "Max Eigenvalue" begin
        celldata = SMatrix{3, 4, Float64}([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ])
        normalidx = 1
        result = TerraDG.max_eigenval(eq, celldata, normalidx)
        @test result ≈ 1.1021708396447196
    end

end