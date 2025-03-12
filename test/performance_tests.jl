using TestSole
using Test
using StatProfilerHTML
using PrettyTables
using BenchmarkTools
using DataFrames
using Logging

Logging.disable_logging(Logging.Info)


function run_benchmark_tests()
    
    results = DataFrame(
        N_Nodes = Int[],
        Depth = Int[],
        Dataset = String[],
        Dropout_Input = Float64[],
        Dropout_Hidden = Float64[],
        Use_Bias = Bool[],
        Execution_Time = Float64[],
        Memory_Allocated = Float64[],
    )

    # Test parameters
    nodes_range = 3:2:21
    depths = 3:5
    datasets = ["iris"]
    dropout_configs = [(1.0, 1.0), (0.5, 0.5), (0.75, 0.75)]
    bias_configs = [false, true]

    for n_nodes in nodes_range
        for depth in depths
            for dataset in datasets
                for (dropout_input, dropout_hidden) in dropout_configs
                    for use_bias in bias_configs
                        params = (
                            n_nodes,
                            depth,
                            dataset,
                            dropout_input,
                            dropout_hidden,
                            use_bias,
                        )
                        @show params

                        b = @benchmark TestSole.experimenter(
                            $n_nodes,
                            $dataset,
                            $depth,
                            $dropout_input,
                            $dropout_hidden,
                            $use_bias;
                            print_progress = false,
                        )

                        push!(
                            results,
                            (
                                params...,
                                median(b.times) / 1e9,  
                                median(b.memory) / 1024 / 1024,  
                            ),
                        )
                    end
                end
            end
        end
    end

    return results
end

@testset "TipsySole.jl Performance Tests" begin
    results = run_benchmark_tests()

    # PrettyTables
    headers =
        ["Nodes" "Depth" "Dataset" "Drop In" "Drop Hidden" "Bias" "Time (s)" "Memory (MB)"]

    pretty_table(
        results,
        header = headers,
        formatters = ft_printf("%.3f", [7, 8]), 
        alignment = [:right, :right, :left, :right, :right, :center, :right, :right],
    )

    @test size(results, 1) > 0  
    @test all(results.Execution_Time .>= 0)  
    @test all(results.Memory_Allocated .>= 0)  
end

function generate_performance_report(results)
    println("\n=== Performance Report ===\n")

    
    for depth in unique(results.Depth)
        println("\nDepth = $depth:")
        subset = results[results.Depth.==depth, :]

        pretty_table(
            select(
                combine(
                    groupby(subset, :N_Nodes),
                    :Execution_Time => mean => :Avg_Time,
                    :Memory_Allocated => mean => :Avg_Memory,
                ),
                [:N_Nodes, :Avg_Time, :Avg_Memory],
            ),
            header = ["Nodes", "Avg Time (s)", "Avg Memory (MB)"],
            formatters = ft_printf("%.3f", [2, 3]),
        )
    end

   
    fastest = argmin(results.Execution_Time)
    println("\nFastest Configuration:")
    pretty_table(results[fastest:fastest, :], formatters = ft_printf("%.3f", [7, 8]))
end


results = run_benchmark_tests()
generate_performance_report(results)
