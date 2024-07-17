@testset "Mass matrix is correct for 2D" begin
    basis_order1 = TerraDG.Basis(1, 2)
    basis_order2 = TerraDG.Basis(2, 2)
    @test TerraDG.massmatrix(basis_order1, 2) == reshape([1], 1, 1)
    massmatrix_order2 = Diagonal([0.25, 0.25, 0.25, 0.25])
    @test TerraDG.massmatrix(basis_order2, 2) == massmatrix_order2
end

@testset "Projection/evaluation works for polynomials" begin
    ns = [1, 2, 3, 4, 5, 6]
    funs = [
        (x, y) -> [1],
        (x, y) -> [x + y],
        (x, y) -> [(x + y)^2],
        (x, y) -> [(x + y)^3],
        (x, y) -> [(x + y)^4],
        (x, y) -> [(x + y)^5]
    ]
    points = [
        [0.21, 0.23],
        [0.38, 0.93],
        [0.92, 0.23],
        [0.01, 0.01],
        [0.99, 0.99],
    ]
    proj_reshaped(basis, func) = reshape(
        TerraDG.project_to_reference_basis(func, basis, 1), length(basis.quadweights)^2)

    for (n, fun) in zip(ns, funs)
        basis = TerraDG.Basis(n, 2)
        coeffs = proj_reshaped(basis, fun)
        for point in points
            evaluated = TerraDG.evaluate_basis(basis, coeffs, point)
            reference = fun(point[1], point[2])[1]
            @test isapprox(evaluated, reference, atol=10e-7)
        end
    end
end

@testset "Derivative matrix is correct" begin
    ns = [1, 2, 3, 2, 3, 4, 5, 6, 2, 5]
    funs = [
        (x, y) -> [1]
        (x, y) -> [1]
        (x, y) -> [1]
        (x, y) -> [x + y]
        (x, y) -> [(x + y)^2]
        (x, y) -> [(x + y)^3]
        (x, y) -> [(x + y)^4]
        (x, y) -> [(x + y)^5]
        (x, y) -> [x * y]
        (x, y) -> [(x^3 + y^4 + x^2 * y^2)]
    ]
    funs_deriv_x = [
        (x, y) -> [0]
        (x, y) -> [0]
        (x, y) -> [0]
        (x, y) -> [1]
        (x, y) -> [2 * (x + y)]
        (x, y) -> [3 * (x + y)^2]
        (x, y) -> [4 * (x + y)^3]
        (x, y) -> [5 * (x + y)^4]
        (x, y) -> [y]
        (x, y) -> [x * (3 * x + 2 * y^2)]
    ]
    funs_deriv_y = [
        (x, y) -> [0]
        (x, y) -> [0]
        (x, y) -> [0]
        (x, y) -> [1]
        (x, y) -> [2 * (x + y)]
        (x, y) -> [3 * (x + y)^2]
        (x, y) -> [4 * (x + y)^3]
        (x, y) -> [5 * (x + y)^4]
        (x, y) -> [x]
        (x, y) -> [2 * y * (x^2 + 2 * y^2)]
    ]

    for (n, fun, deriv_x, deriv_y) in zip(ns, funs, funs_deriv_x, funs_deriv_y)
        proj_reshaped(basis, func) = reshape(
            TerraDG.project_to_reference_basis(func, basis, 1), length(basis.quadweights)^2)
        basis = TerraDG.Basis(n, 2)
        fun_evaluated = proj_reshaped(basis, fun)
        reference_derivx_evaluated = proj_reshaped(basis, deriv_x)
        reference_derivy_evaluated = proj_reshaped(basis, deriv_y)
        ∇ = TerraDG.derivativematrix(basis)
        derivx_evaluated = transpose(∇[:, 1:length(basis)]) * fun_evaluated
        derivy_evaluated = transpose(∇[:, length(basis)+1:end]) * fun_evaluated

        @test all(isapprox.(derivx_evaluated, reference_derivx_evaluated, atol=10e-5))
        @test all(isapprox.(derivy_evaluated, reference_derivy_evaluated, atol=10e-5))
    end
end

@testset "Face projection matrix is correct" begin
    ns = [1, 2, 3, 4, 5, 6]
    funs = [
        (x, y) -> [1, 1, 1]
        (x, y) -> [0, 0, 0]
        (x, y) -> [0, 1, 0]
    ]
    for n in ns
        basis = TerraDG.Basis(n, 2)
        for func in funs
            func_proj = TerraDG.project_to_reference_basis(func, basis, 3)
            for face in [TerraDG.top, TerraDG.bottom, TerraDG.left, TerraDG.right]
                fp = TerraDG.face_projection_matrix(basis, face)
                func_proj_face = fp * func_proj
                @test sum(func_proj) / length(func_proj) ≈ sum(func_proj_face) / length(func_proj_face) atol=10e-7
            end
        end
    end
end
