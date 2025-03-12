using TestSole
using Test
using Random
using TestSole: TritVector, BalancedTritVector, BalancedTernaryVector

function run_test_suite(suite_name, test_files)
    @testset "$suite_name" begin
        for test_file in test_files
            println("Running test: $test_file")
            include(test_file)
        end
    end
end

println("Running tests on Julia version: ", VERSION)
println("="^50)


@testset "TestSole.jl" begin

    @testset "Data Structures" begin
        run_test_suite(
            "TritVector",
            ["trit-vector.jl", "balanced-trit-vector.jl", "balanced-ternary-vector.jl"],
        )
    end

    @testset "Main Functionality" begin
        include("main.jl")
        include("twoleveldnfformula.jl")
    end

    @testset "Performance Tests" begin
        include("performance_tests.jl")
        include("performance_generate_combination.jl")
    end

end

println("\n", "="^50)
println("All tests completed.")
