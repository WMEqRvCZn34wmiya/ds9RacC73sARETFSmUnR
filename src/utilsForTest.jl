using Pkg
Pkg.activate(".")
using Revise
using Random
using Logging
using Dates
using DataStructures
using ScikitLearn
using SoleModels
using DecisionTree: load_data, build_forest, apply_forest
using AbstractTrees
using SoleData
using SoleData: UnivariateScalarAlphabet

using ModalDecisionTrees
using SoleLogics
using DataFrames
using BenchmarkTools, StatProfilerHTML
using Base: intersect
using Base.Threads: @threads
using Base.Threads: Atomic, atomic_add!
using Profile
using ConcurrentCollections
using ProgressMeter
using DelimitedFiles
using StatsBase

include("lumen/main.jl")
using SolePostHoc: BATrees
using SolePostHoc: intrees
using SolePostHoc: REFNE
using SolePostHoc: TREPAN

include("data.jl")
###########################################################################################################
#                                       Experimenter                                                      #
###########################################################################################################

"""
    learn_and_convert(
        numero_alberi::Int,
        nome_dataset::String,
        max_depth::Int = -1,
    )

Trains a random forest model on the specified dataset and returns the trained model.

Args:
    numero_alberi (Int): The number of trees to train in the random forest.
    nome_dataset (String): The name of the dataset to use for training.
    max_depth (Int): The maximum depth of the decision trees in the random forest. Defaults to -1, which means no maximum depth.

Returns:
    The trained random forest model.
"""
function learn_and_convert(
    numero_alberi::Int,
    nome_dataset::String,
    max_depth::Int=-1,
)
    start_time = time()
    println(
        "\n\n$COLORED_TITLE$TITLE\n PART 0 DATASET CONFIGURATION \n$TITLE$RESET",
    )

    println("\n\n$COLORED_INFO$TITLE\n HARDCODED MODE \n$TITLE$RESET")

    supported_datasets = ["iris", "zoo", "monks-1","monks-2","monks-3", "house-votes", "balance-scale", "hayes-roth", "primary-tumor", "soybean-small",
                          "tictactoe", "car", "tae", "cmc","heart","penguins","glass","mushroom","lenses","lymphography","hepatitis",
                          "bean","madman","post-operative","urinary-d1","urinary-d2"]
    if !(nome_dataset in supported_datasets)
        error("Dataset $nome_dataset not supported. Available datasets: $supported_datasets")
    end

    features, labels, features_train, labels_train, features_test, labels_test = load_data_hardcoded(nome_dataset)


    @info StatsBase.countmap(labels)

    @info "dataset loaded: $(nome_dataset) correctly... good luck!"

    println(
        "\n\n$COLORED_TITLE$TITLE\n PART 1 GENERATION OF THE FOREST with decisionTree.jl \n$TITLE$RESET",
    )

    # set of classification parameters and respective default values

    # n_subfeatures: number of features to consider randomly for each split (default: -1, sqrt(# features))
    # n_trees: number of trees to train (default: 10)
    # partial_sampling: fraction of samples to train each tree on (default: 0.7)
    # max_depth: maximum depth of the decision trees (default: no maximum (-1))
    # min_samples_leaf: minimum number of samples each leaf must have (default: 5)
    # min_samples_split: minimum number of samples required for a split (default: 2)
    # min_purity_increase: minimum purity required for a split (default: 0.0)
    # keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
    # multi-threaded forests must be initialized with an `Int`

    n_subfeatures = -1
    n_trees = numero_alberi
    partial_sampling = 0.7
    #max_depth = -1              # from 6 it becomes too much... already with 1...
    min_samples_leaf = 5
    min_samples_split = 2
    min_purity_increase = 0.0
    seed = 202

    model = build_forest(
        labels_train,
        features_train,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase;
        rng=seed,
    )


    println(model)

    println(
        "\n\n$COLORED_TITLE$TITLE\n PART 2 CONVERSION OF THE FOREST INTO SOLE with solemodel variant \n$TITLE$RESET",
    )

    f = solemodel(model) # f = solemodel(model; classlabels = labels)
    println(f)

    return f, model, start_time, features_train, labels_train, features_test, labels_test, features, labels
end


"""
    run_analysis_on_datasets()

Run an analysis on a set of supported datasets using different numbers of trees and parameter values.

# Supported Datasets
- "iris"
- "monks"
- "balance-scale"
- "hayes-roth"
- "car"

# Tree Numbers
- 3
- 5
- 7

# Parameter Values
- 0.1 to 1.0 (in increments of 0.1)

# Description
For each dataset and each specified number of trees, this function trains a forest model and evaluates it using different parameter values. 
The results, including rule evaluations and execution times, are saved in the `results` directory.

# Output
- A dictionary containing the rule evaluations for each dataset, number of trees, and parameter combination.
- A dictionary containing the execution times for each dataset, number of trees, and parameter combination.

# Example
```julia
results, execution_times = run_analysis_on_datasets()
"""
function run_analysis_on_datasets()
    # Initial configuration
    supported_datasets = ["iris", "monks", "balance-scale", "hayes-roth", "car"]
    param_values = collect(1.0)
    tree_numbers = [3]#, 5, 7] # , 10
    results = Dict()
    execution_times = Dict()

    !isdir("results") && mkdir("results")

    for dataset_name in supported_datasets
        results[dataset_name] = Dict()
        execution_times[dataset_name] = Dict()
        println("\nDataset analysis: $dataset_name")

        for num_trees in tree_numbers
            println("\nNumber of trees: $num_trees")


            f, model, start_time, features_train, labels_train, features_test, labels_test, features, labels = learn_and_convert(num_trees, dataset_name, 3)

            if isnothing(model) || isnothing(f)
                @warn "Model or function not initialized for $dataset_name with $num_trees trees"
                continue
            end

            X, y = begin
                #features, labels = load_data_hc(dataset_name)
                features = float.(features_test)
                labels = string.(labels_test)
                features, labels
            end

            if isempty(features) || any(isnan, features)
                @warn "Invalid data for $dataset_name"
                continue
            end

            data_df = DataFrame(features, :auto)

            for param1 in param_values
                for param2 in param_values
                    println("\nParameters: $param1, $param2")

                    local ds, y_pred, rule_evaluations, rule_evaluations_unminimized, tuplevector, execution_time, consequents
                    try
                        start_execution = time()
                        ds, Lumen_info = lumen(
                            model,
                            start_time=start_time,
                            vertical=param1,
                            horizontal=param2,
                            ott_mode=false,
                            controllo=false,
                            solemodel=f;
                            silent=false,
                            apply_function=apply_forest,
                        )
                        execution_time = time() - start_execution
                        tuplevector = Lumen_info.vectPrePostNumber
                        unminimized_ds = Lumen_info.unminimized_ds
                        consequents = [rule.consequent for rule in ds.rules]

                        println("Lumen execution time for $dataset_name (trees: $num_trees, param1: $param1, param2: $param2): $(round(execution_time, digits=2)) seconds")

                        execution_times[dataset_name][(num_trees, param1, param2)] = execution_time

                        y_pred = apply(
                            f,
                            SoleData.scalarlogiset(data_df; allow_propositional=true)
                        )

                        if isempty(y_pred) || any(isnothing, y_pred)
                            @warn "Invalid predictions"
                            continue
                        end

                        eval_data = scalarlogiset(
                            DataFrame(X, ["V$(i)" for i in 1:size(X, 2)]);
                            allow_propositional=true
                        )

                        # Move these assignments inside the try block
                        rule_evaluations = map(r -> SoleModels.evaluaterule(r, eval_data, y_pred; compute_explanations=true), SoleModels.rules(ds))
                        rule_evaluations_unminimized = map(r -> SoleModels.evaluaterule(r, eval_data, y_pred; compute_explanations=true), SoleModels.rules(unminimized_ds))

                    catch e
                        @warn "Error during processing" dataset = dataset_name trees = num_trees params = (param1, param2) error = e
                        continue
                    end

                    try
                        results[dataset_name][(num_trees, param1, param2)] = (rule_evaluations, rule_evaluations_unminimized)

                        filename = "results/$(dataset_name)_$(num_trees)_$(param1)_$(param2).txt"
                        open(filename, "w") do io
                            println(
                                io,
                                """madman
                                Dataset: $dataset_name
                                Number of trees: $num_trees
                                Parameter 1: $param1
                                Parameter 2: $param2
                                Execution time: $(round(execution_time, digits=2)) seconds
                                Timestamp: $(Dates.now())

                                Rule evaluation:
                                """
                            )

                            for (rule_evaluations_name, _r) in [("rule_evaluations", rule_evaluations), ("rule_evaluations_unminimized", rule_evaluations_unminimized)]
                                println(io, "\n=== $(rule_evaluations_name == "rule_evaluations" ? "Minimized" : "Unminimized") Rules ===")
                                println(io, "=" * repeat("-", 40))

                                for (i, eval) in enumerate(_r)
                                    println(io, "\nRule $(consequents[i]):")
                                    println(io, "▸ Pattern: ", tuplevector[i])
                                    println(io, "▸ Evaluation: ", eval)

                                    natoms_expl = map(expls -> let
                                            x = natoms.(expls)
                                            length(unique(x)) == 1 ?
                                            (minimum(x), maximum(x), mean(x), 0.0) :
                                            (minimum(x), maximum(x), mean(x), std(x))
                                        end, filter(!isempty, eval.explanations))

                                    println(io, "▸ Explanation Statistics:")
                                    println(io, "  - Min-Avg: ", round(StatsBase.mean(map(x -> x[1], natoms_expl)), digits=5))
                                    println(io, "  - Min-Std: ", round(StatsBase.std(map(x -> x[1], natoms_expl)), digits=5))
                                    println(io, "  - Maximum: ", round(StatsBase.mean(map(x -> x[2], natoms_expl)), digits=5))
                                    println(io, "  - Average: ", round(StatsBase.mean(map(x -> x[3], natoms_expl)), digits=5))
                                    println(io, "  - Std Dev: ", round(StatsBase.mean(map(x -> x[4], natoms_expl)), digits=5))
                                    println(io, repeat("-", 40))
                                end
                            end
                        end
                    catch e
                        @warn "Error saving results" dataset = dataset_name error = e
                    end
                end
            end
        end
    end
    return results, execution_times
end


"""
    calculate_forest_accuracy_ff(f)

Calculate the accuracy of a forest model using test data.

# Arguments
- `f`: The forest model to be evaluated.

# Description
This function reads the test data from `src/owncsv/X_test.csv` and `src/owncsv/y_test.csv`, applies the forest model `f` to the test features, and calculates the accuracy of the predictions against the true labels.

# Output
- The accuracy of the forest model on the test data.

# Example
```julia
accuracy = calculate_forest_accuracy_ff(f)
"""
function calculate_forest_accuracy_ff(f)
    x_data = DelimitedFiles.readdlm("src/owncsv/X_test.csv", ',')
    y_data = DelimitedFiles.readdlm("src/owncsv/y_test.csv", ',')

    features = x_data[2:end, :]
    labels = vec(y_data[2:end])

    # Convert features to Float64 and handle any missing values
    features = map(x -> ismissing(x) ? 0.0 : Float64(x), features)

    # Ensure labels are strings and handle potential edge cases
    labels = map(x -> ismissing(x) ? "0" : string(Int(round(parse(Float64, string(x))))), labels)

    # Create DataFrame with proper column names
    df = DataFrame(features, :auto)

    y_pred = apply(
        f,
        SoleData.scalarlogiset(
            df;
            allow_propositional=true,
        ),
    )

    acc = SoleModels.accuracy(labels, y_pred)
    return acc
end

"""
    calculate_forest_accuracy(f)

Calculate the accuracy of a forest model using test data.

# Arguments
- `f`: The forest model to be evaluated.

# Description
This function reads the test data from `src/owncsv/X_test.csv` and `src/owncsv/y_test.csv`, applies the forest model `f` to the test features, and calculates the accuracy of the predictions against the true labels.

# Output
- The accuracy of the forest model on the test data.

# Example
```julia
accuracy = calculate_forest_accuracy(f)
"""
function calculate_forest_accuracy(f,features_test,labels_test)
    x_data = DelimitedFiles.readdlm("src/owncsv/X_test.csv", ',')
    y_data = DelimitedFiles.readdlm("src/owncsv/y_test.csv", ',')

    #features = x_data[2:end, :]
    #labels = y_data[2:end, 1]
    #features = float.(features)
    #labels = string.(labels)

    y_pred = apply(
        f,
        SoleData.scalarlogiset(
            DataFrame(features_test, :auto);
            allow_propositional=true,
        ),
    )

    acc = SoleModels.accuracy(labels_test, y_pred)
    println("Accuracy: ", acc)
    return acc
end



"""
    run_forest_accuracy_analysis()

Run an accuracy analysis on a set of supported datasets using different numbers of trees in the forest.

# Supported Datasets
- "iris"
- "monks"
- "balance-scale"
- "hayes-roth"
- "car"

# Tree Numbers
- 3
- 5
- 7

# Description
For each dataset and each specified number of trees, this function trains a forest model and calculates its accuracy. 
The results are saved in the `results/forest_accuracies.txt` file.

# Special Handling
- The "monks" dataset is loaded with special handling and without a maximum depth constraint.

# Output
The function writes the accuracy results to `results/forest_accuracies.txt`.

# Example
```julia
run_forest_accuracy_analysis()
"""
function run_forest_accuracy_analysis()
    supported_datasets = ["iris", "monks", "balance-scale", "hayes-roth", "car"]
    tree_numbers = [3]#[3, 5, 7] # , 10

    !isdir("results") && mkdir("results")

    open("results/forest_accuracies.txt", "w") do io
        println(io, "Forest Accuracy Analysis\n", "="^30)

        for dataset_name in supported_datasets
            println(io, "\nDataset: $dataset_name")
            println(io, "-"^50)

            for num_trees in tree_numbers
                println(io, "\nNumber of Trees: $num_trees")

                try
                    if dataset_name == "monks"
                        println(io, "Loading monks dataset with special handling...")
                        f, model, _ = learn_and_convert(num_trees, dataset_name, -1) # Removing max_depth constraint
                    else
                        f, model, _ = learn_and_convert(num_trees, dataset_name, 3)
                    end

                    if isnothing(model) || isnothing(f)
                        println(io, "ERROR: Model initialization failed")
                        continue
                    end
                    if dataset_name == "monks"
                        accuracy = calculate_forest_accuracy_ff(f)
                    else
                        accuracy = calculate_forest_accuracy(f)
                    end

                    println(io, "Accuracy: $accuracy")

                catch e
                    println(io, "Processing error: $e")
                end
            end
        end
    end
end