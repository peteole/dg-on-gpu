using YAML
using StaticArrays

@testset "Equations" begin

    @testset "Advection" begin
        include("./test_advection.jl")
    end

    @testset "Acoustic Gaussian Wave" begin
        include("./test_acoustic.jl")
    end

    @testset "Acoustic Constant Wave" begin
        include("./test_acoustic_constant.jl")
    end

    @testset "Sibson" begin
        include("./test_sibson.jl")
    end

    @testset "Euler Sod Schock Tube" begin
        include("./test_euler_sod_shock_tube.jl")
    end

    @testset "Euler Gaussian Wave" begin
        include("./test_euler_gaussian_wave.jl")
    end

    @testset "Test advection is running" begin
        try
            TerraDG.main("./configs/advection.yaml")
            @test true
        catch e
            @test false
        end
    end

    @testset "Test sibson is running" begin
        try
            TerraDG.main("./configs/interpolation.yaml")
            @test true
        catch e
            @test false
        end
    end

    @testset "Test acousticwave is running" begin
        try
            TerraDG.main("./configs/acousticwave.yaml")
            @test true
        catch e
            @test false
        end
    end

    @testset "Test constantwave is running" begin
        try
            TerraDG.main("./configs/constantwave.yaml")
            @test true
        catch e
            @test false
        end
    end

    @testset "Test euler_gaussian_wave is running" begin
        try
            TerraDG.main("./configs/euler_gaussian_wave.yaml")
            @test true
        catch e
            @test false
        end
    end

    @testset "Test euler_sod_shock_tube is running" begin
        try
            TerraDG.main("./configs/euler_sod_shock_tube.yaml")
            @test true
        catch e
            @test false
        end
    end

end
