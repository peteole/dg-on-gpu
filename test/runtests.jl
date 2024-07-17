using TerraDG
using Test
using LinearAlgebra

@testset "All tests" begin

    @testset "Grid" begin
        include("test_grid.jl")
    end

    @testset "Polynomials" begin
        include("test_polynomials.jl")
    end

    @testset "Basis" begin
        include("test_basis.jl")
    end

    @testset "Equations" begin
        include("./test_equations/test_equations.jl")
    end

    @testset "Factorizations" begin
        include("test_sum_factorizations.jl")
    end

end
