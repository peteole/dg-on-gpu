@testset "Quadrature on the reference element" begin
    n = 5
    basis = TerraDG.Basis(n, 1)
    p = basis.quadpoints
    wp = basis.quadweights
    func = (x, y) -> x * y * y
    sum = 0.0
    
    for i in CartesianIndices((n, n))
        sum += func(p[i[1]], p[i[2]]) * wp[i[1]] * wp[i[2]]
    end

    @test abs(sum - 1.0 / 6.0) <= 10e-8
end

@testset "Maps reference coordinate to correct physical coord" begin
    cellcenter = (1.0, 1.0)
    cellsize = (0.5, 0.5)
    
    cell = TerraDG.Cell(cellcenter,
        cellsize,
        (4, 8, 6, 2),
        (TerraDG.regular, TerraDG.regular, TerraDG.regular, TerraDG.regular),
        5)

    @test TerraDG.globalposition(cell, [0.5, 0.5]) == collect(cellcenter)
    @test TerraDG.globalposition(cell, [0.0, 0.0]) == [0.75, 0.75]
    @test TerraDG.globalposition(cell, [1.0, 1.0]) == [1.25, 1.25]
    @test_throws BoundsError TerraDG.globalposition(cell, [1.1, 0.0])
    @test_throws BoundsError TerraDG.globalposition(cell, [-0.1, 0.0])
end

@testset "Global to reference to global is same" begin
    cellcenter = (1.0, 1.0)
    cellsize = (0.5, 0.5)
    cell = TerraDG.Cell(cellcenter,
        cellsize,
        (4, 8, 6, 2),
        (TerraDG.regular, TerraDG.regular, TerraDG.regular, TerraDG.regular),
        5)
    coords = [
        [0.5, 0.5],
        [0.0, 0.0],
        [1.0, 1.0]
    ]

    for coord ∈ coords
        global_coord = TerraDG.globalposition(cell, coord)
        local_coord = TerraDG.localposition(cell, global_coord)
        @test local_coord == coord
    end
end


@testset "Volume of 2D cells is correct" begin
    sizes = [
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5)
    ]
    volumes = [
        1.0,
        4.0,
        0.25
    ]

    for (size, volume) ∈ zip(sizes, volumes)
        cell = TerraDG.Cell((0.5, 0.5),
            size,
            (4, 8, 6, 2),
            (TerraDG.regular, TerraDG.regular, TerraDG.regular, TerraDG.regular),
            5)
        @test TerraDG.volume(cell) == volume
    end
end

@testset "Area of 2D cells is correct" begin
    sizes = [
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5)
    ]
    areas = [
        1.0,
        2.0,
        0.5
    ]

    for (size, area) ∈ zip(sizes, areas)
        cell = TerraDG.Cell((0.5, 0.5),
            size,
            (4, 8, 6, 2),
            (TerraDG.regular, TerraDG.regular, TerraDG.regular, TerraDG.regular),
            5)
        @test TerraDG.area(cell) == area
    end
end
