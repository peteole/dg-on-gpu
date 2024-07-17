**Running Tests**

    ] test

**Coverage**

Use julia REPL (Terminal).

    JULIA_REVISE_POLL=1 julia --project=. --threads=16

    using Pkg
    Pkg.test("TerraDG"; coverage=true)

    include("test/coverage.jl")

To create coverage HTML report

    brew install lcov
    genhtml -o coverage/html coverage/lcov.info

To delete all .cov files

    find . -name "*.cov" -type f -delete
