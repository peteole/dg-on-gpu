function is_gpu_available()
    try
        CUDA.has_cuda() && CUDA.functional()
    catch
        false
    end
end

array_type = is_gpu_available() ? CuArray : Array

@testset "Advection Equation" begin
    configfile = "./configs/advection.yaml"
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
        @test result == @SVector [sin(2π * (0.5 + 0.5)), sin(2π * 0.5), 1.0]
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
        ndofs = 3
        celldofs = SMatrix{basis_size_nd, ndofs, Float64}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        cellflux = zeros(MMatrix{2*basis_size_nd, ndofs, Float64})
        TerraDG.evaluate_flux(eq, celldofs, cellflux)
        expected_cellflux = SMatrix{2 * basis_size_nd, ndofs, Float64}((1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0))
        @test cellflux == expected_cellflux
    end

    @testset "Max Eigenvalue" begin
        result = TerraDG.max_eigenval(eq, :some_celldata, 1)
        @test result == 1
    end

    @testset "Grid" begin
        result = grid = TerraDG.make_grid(config, eq, scenario, array_type, typeof(config.end_time))
        @test isa(grid, TerraDG.Grid)
        @test grid.size == (1.0, 1.0)
        @test grid.maxeigenval == -1.0
        @test grid.time == 0.0
        @test length(grid.cells) == config.grid_elements^2

        cells_cpu = Array(grid.cells)
        
        cell = cells_cpu[1]
        @test isa(cell, TerraDG.Cell)
        @test cell.center == (0.025, 0.025)
        @test cell.size == (0.05, 0.05)
        @test cell.neighbor_indices == (20, 21, 2, 381)
        @test cell.facetypes == (TerraDG.regular, TerraDG.regular, TerraDG.regular, TerraDG.regular)
        @test cell.dataidx == 1

        cell_boundary = cells_cpu[end]
        @test cell_boundary.center == (0.975, 0.975)
        @test cell_boundary.size == (0.05, 0.05)
        @test cell_boundary.neighbor_indices == (399, 20, 381, 380)
        @test cell_boundary.facetypes == (TerraDG.regular, TerraDG.regular, TerraDG.regular, TerraDG.regular)
        @test cell_boundary.dataidx == length(grid.cells)
    end

end