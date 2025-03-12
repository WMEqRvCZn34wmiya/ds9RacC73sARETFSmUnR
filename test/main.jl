using TestSole
using Test
using StatProfilerHTML
using PrettyTables
using BenchmarkTools
using DataFrames

@testset "Main Tests" begin
    @testset "Basic Functionality" begin
        @test_nowarn TestSole.experimenter(3, "iris", 3)
    end

    @testset "Parameter Variations" begin
        @test_nowarn TestSole.experimenter(5, "iris", 4, 0.5, 0.5)
        @test_nowarn TestSole.experimenter(7, "iris", 3, 0.75, 0.75, true, true)
    end
end
