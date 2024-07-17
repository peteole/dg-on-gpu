@testset "Lagrange polynomials are correct" begin
    δ(a, b) =
        if (a == b)
            1.0
        else
            0.0
        end
    for n in range(1, length=6)
        basis = TerraDG.Basis(n, 1)
        points = basis.quadpoints
        for (i, p) in enumerate(points)
            for (j, q) in enumerate(points)
                @test TerraDG.lagrange_1d(points, i, q) == δ(i, j)
            end
        end
    end
end

@testset "Lagrange polynomials sum to one" begin
    for n in range(1, length=6)
        for x in [0.0, 0.5, 1.0]
            basis = TerraDG.Basis(n, 1)
            points = basis.quadpoints
            sum = 0.0
            for (i, p) in enumerate(points)
                sum += TerraDG.lagrange_1d(points, i, x)
            end
            @test isapprox(sum, 1, atol=10e-7)
        end
    end
end

@testset "Integral of derivatives of 1D-Lagrange polynomials is correct" begin
    for L in 0.2:0.2:1
        for n in range(1, length=6)
            basis = TerraDG.Basis(n, 1)
            quadrature_points = basis.quadpoints .* L
            quadrature_weights = basis.quadweights .* L
            for i in 1:n
                ϕ_i(x) = TerraDG.lagrange_1d(quadrature_points, i, x)
                ∂xϕ_i(x) = TerraDG.lagrange_diff(quadrature_points, i, x)
                @test isapprox(ϕ_i(L) - ϕ_i(0),
                    sum(quadrature_weights[m] * ∂xϕ_i(quadrature_points[m]) for m in eachindex(quadrature_points); init=0.0), atol=10e-7)
            end
        end
    end
end

@testset "Integral of derivatives of 2D-Lagrange polynomials is also correct" begin
    for L in 0.2:0.2:1
        for n in range(1, length=6)
            basis = TerraDG.Basis(n, 2)
            quadrature_points_d1 = basis.quadpoints .* L
            quadrature_weights_d1 = basis.quadweights .* L
            quadrature_points_d2 = basis.quadpoints
            quadrature_weights_d2 = basis.quadweights

            for i in 1:n
                for j in 1:n
                    # Test the polynomial Φ_ij(x, y) = ϕ_i(x)ϕ_j(y)
                    # where ϕ_i(x) is the i-th Lagrange polynomial in 1D
                    # Φ_ij is defined on the reference element [0, L] x [0, 1]
                    # ϕ_i(x_j) = δ_ij if x_j is the j-th quadrature point
                    # ∂_xk Φ_ij(xk, y) = ∂_xk (ϕ_i(xk)ϕ_j(y)) = (∂_xk ϕ_i(xk))ϕ_j(y)
                    Φ_ij(x,y) = TerraDG.lagrange_1d(quadrature_points_d1, i, x) * TerraDG.lagrange_1d(quadrature_points_d2, j, y)
                    ∂_xΦ_ij(x,y) = TerraDG.lagrange_diff(quadrature_points_d1, i, x) * TerraDG.lagrange_1d(quadrature_points_d2, j, y)
                    @test isapprox(
                        sum(quadrature_weights_d2[m] * (Φ_ij(L, quadrature_points_d2[m])-Φ_ij(0, quadrature_points_d2[m])) for m in eachindex(quadrature_points_d2); init=0.0),
                        sum(quadrature_weights_d1[m1]*quadrature_weights_d2[m2]*∂_xΦ_ij(quadrature_points_d1[m1],quadrature_points_d2[m2]) for m1 in eachindex(quadrature_points_d1) for m2 in eachindex(quadrature_points_d2); init=0.0), atol=10e-7)
                end
            end
            for i in 1:n
                for j in 1:n
                    # Test the polynomial Φ_ij(x, y) = ϕ_i(x)ϕ_j(y)
                    # where ϕ_i(x) is the i-th Lagrange polynomial in 1D
                    # Φ_ij is defined on the reference element [0, L] x [0, 1]
                    # ϕ_i(x_j) = δ_ij if x_j is the j-th quadrature point
                    # ∂_yk Φ_ij(x, yk) = ∂_xk (ϕ_i(xk)ϕ_j(y)) = ϕ_i(x)(∂_yk ϕ_j(y))
                    Φ_ij(x,y) = TerraDG.lagrange_1d(quadrature_points_d1, i, x) * TerraDG.lagrange_1d(quadrature_points_d2, j, y)
                    ∂_yΦ_ij(x,y) = TerraDG.lagrange_1d(quadrature_points_d1, i, x) * TerraDG.lagrange_diff(quadrature_points_d2, j, y)
                    @test isapprox(
                        sum(quadrature_weights_d1[m] * (Φ_ij(quadrature_points_d1[m],1)-Φ_ij(quadrature_points_d1[m],0)) for m in eachindex(quadrature_points_d2); init=0.0),
                        sum(quadrature_weights_d1[m1]*quadrature_weights_d2[m2]*∂_yΦ_ij(quadrature_points_d1[m1],quadrature_points_d2[m2]) for m1 in eachindex(quadrature_points_d1) for m2 in eachindex(quadrature_points_d2); init=0.0), atol=10e-7)
                end
            end
        end
    end
end

@testset "Derivative of const = 0" begin
    for n = 1:6
        basis = TerraDG.Basis(n, 1)
        for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            @test isapprox(
                sum([TerraDG.lagrange_diff(basis.quadpoints, i, x) for i = 1:size(basis, 1)]), 0, atol=10e-6)
        end
    end
end
