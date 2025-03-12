using Pkg
Pkg.activate(".")
using Revise
using DataFrames
using ComplexityMeasures
using DecisionTree
# Include all files at the beginning to avoid world age problems
include("utilsForTest.jl")
include("suitfortest.jl")
include("apiIntrees.jl")
include("apiRefne.jl")
include("RULECOSIplus.jl")



for c in [1.0]

    datasets = ["monks-1", "monks-2", "monks-3", "hayes-roth", "balance-scale"]

    # Create CSV file directly and write a more detailed header
    open("evaluation_results_experiment1.csv", "w") do csv_file
        # Write header with all the metrics you want
        println(csv_file, "Dataset,Algorithm,Number_of_Terms,Execution_Time,Rule_ID,Sensitivity,Specificity,Min_Avg,Min_Std,Maximum,Average,Std_Dev,NumAtoms")

        #======================================================================================================================================
                                                        SETUP ALBERI X Y ECC...
        ======================================================================================================================================#
        for dataset in datasets
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n ADDESTRAMENTO ALBERO $dataset \n$TITLE$RESET",
            )

            f, model, start_time, features_train, labels_train, features_test, labels_test, features, labels = learn_and_convert(10, dataset, 3)

            accuracy = calculate_forest_accuracy(f, features_test, labels_test)
            println("Model accuracy: $(round(accuracy * 100, digits=2))%")

            println("importance vector:", create_importance_vector(impurity_importance(model)))

            #======================================================================================================================================
                                                                    LUMEN
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n LUMEN \n$TITLE$RESET",
            )

            timelumen = @elapsed begin 
                dslumen, Lumeninfo = lumen(
                    model;
                    start_time=start_time,
                    vertical=1.0,
                    horizontal=c,
                    ott_mode=false,
                    controllo=false,
                    solemodel=f,
                    silent=false,
                    apply_function=apply_forest,
                    vetImportance=create_importance_vector(impurity_importance(model))
                )
            end

            println("Execution time: $timelumen seconds")

            #======================================================================================================================================
                                                                    BATREES
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n BATREES \n$TITLE$RESET",
            )

            timebatrees = @elapsed begin
                dsbatrees = BATrees.batrees(f)
            end

            println("Execution time: $timebatrees seconds")

            #======================================================================================================================================
                                                                    INTREES
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n INTREES \n$TITLE$RESET",
            )

            x = features_train
            y = labels_train

            X = SoleData.scalarlogiset(DataFrame(x, :auto))

            if dataset == "balance-scale"
                timeintrees = @elapsed begin
                    dl = intrees(
                        model,
                        X,
                        y,
                        max_rules=10
                    )
                end

            elseif dataset == "monks-2"
                timeintrees = @elapsed begin
                    dl = intrees(
                        model,
                        X,
                        y,
                        max_rules=20
                    )
                end


            elseif dataset == "monks-3"
                timeintrees = @elapsed begin
                    dl = intrees(
                        model,
                        X,
                        y,
                        max_rules=20
                    )
                end

            elseif dataset == "post-operative"
                timeintrees = @elapsed begin
                    dl = intrees(
                        model,
                        X,
                        y,
                        max_rules=55
                    )
                end

            else                              
                timeintrees = @elapsed begin
                    dl = intrees(
                        model,
                        X,
                        y,
                    )
                end
            end
            println("=============")
            println(dl)
            println("=============")

            ll = listrules(dl, use_shortforms=false) # decision list to list of rules
            rules_obj = convert_classification_rules(ll)

            dsintrees = DecisionSet(rules_obj)

            println("Execution time: $timeintrees seconds")


            #======================================================================================================================================
                                                                    REFNE
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n REFNE \n$TITLE$RESET",
            )

            X = features_train
            y = labels_train

            rangeXmin = []
            rangeXmax = []
            for i in 1:size(X, 2)  
                append!(rangeXmax, maximum(X[:, i]))
                append!(rangeXmin, minimum(X[:, i]))

            end

            println("rangeXmin: ", rangeXmin)
            println("rangeXmax: ", rangeXmax)

            timerefne = @elapsed begin
                nf = REFNE.refne(f, rangeXmin, rangeXmax, L=3, max_depth=1000) # L=10 is not every possible ... maybe with 3 is better
            end

            dsrefne = convertApi(nf)
            println("Execution time: $timerefne seconds")

            #======================================================================================================================================
                                                                    TREPAN
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n TREPAN \n$TITLE$RESET",
            )

            # Removed include("utilsForTest.jl") and include("apiRefne.jl") from here

            X = features_train

            # Calculate the number of random samples to add (20% additional compared to X)
            n_samples_original = size(X, 1)
            n_random = round(Int, n_samples_original * 0.16)

            
            random_features = rand(n_random, size(X, 2))

            
            X_combined = vcat(X, random_features)

            timetrepan = @elapsed begin
                nf = TREPAN.trepan(
                    f,
                    X_combined,
                    max_depth=-1,                  # Unlimited depth for a single tree
                    n_subfeatures=1.0,             # Use all available features
                    partial_sampling=1.0,          # Use all samples
                    min_samples_leaf=1,            # Allow leaves with a single sample
                    min_samples_split=2,           # Minimum value required for a split
                    min_purity_increase=5.0e-324,  # Lowest value to allow more splits
                    seed=100,                      # Keep the seed for reproducibility
                )
            end

            dstrepan = convertApi(nf)

            println("Execution time: $timetrepan seconds")

            #======================================================================================================================================
                                                                    RULECOSIPLUS
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n RULE COSI+ \n$TITLE$RESET",
            )

            x = features_train
            y = labels_train

            timerulecosiplus = @elapsed begin
                dl = rulecosiplus(f, x, y) # decision list   
            end

            ll = listrules(dl, use_shortforms=false) # decision list to list of rules

            rules_obj = convert_classification_rules(ll)
            dsrulecosiplus = DecisionSet(rules_obj)

            println("Execution time: $timerulecosiplus seconds")

            #======================================================================================================================================
                                                                    PRINT ALL DS
            ======================================================================================================================================#
            println(
                "\n\n$COLORED_ULTRA_OTT$TITLE\n PRINT ALL DS \n$TITLE$RESET",
            )

            println("Lumen: \n", dslumen)
            println("BATrees: \n", dsbatrees)
            println("Intrees: \n", dsintrees)
            println("Refne: \n", dsrefne)
            println("Trepan: \n", dstrepan)
            println("RuleCosi+: \n", dsrulecosiplus)

            #======================================================================================================================================
                                                                    Evaluate all DS
            ======================================================================================================================================#
            # Create a dictionary to store execution times
            execution_times = Dict(
                "Lumen" => timelumen,
                "BATrees" => timebatrees,
                "Intrees" => timeintrees,
                "Refne" => timerefne,
                "Trepan" => timetrepan,
                "RuleCosiPlus" => timerulecosiplus
            )

            open("output_$dataset.txt", "w") do output_file
                # Define algorithm configurations
                algorithms = [
                    ("Lumen", dslumen),
                    ("BATrees", dsbatrees),
                    ("Intrees", dsintrees),
                    ("Refne", dsrefne),
                    ("Trepan", dstrepan),
                    ("RuleCosiPlus", dsrulecosiplus)
                ]

                # Prepare test data
                X, y = begin
                    features_test = float.(features_test)
                    labels_test = string.(labels_test)
                    features_test, labels_test
                end

                # Common print formatting
                println(output_file, "\n\n$COLORED_ULTRA_OTT$TITLE\n EVALUATE ALL ALGORITHMS \n$TITLE$RESET")
                println("\n\n$COLORED_ULTRA_OTT$TITLE\n EVALUATE ALL ALGORITHMS \n$TITLE$RESET")  

                # Create logiset from test features
                logiset = scalarlogiset(
                    DataFrame(X, ["V$(i)" for i in 1:size(X, 2)]);
                    allow_propositional=true
                )

                # Predict
                y_pred = apply(
                    f,
                    SoleData.scalarlogiset(
                        DataFrame(features_test, :auto);
                        allow_propositional=true,
                    )
                )

                # Unified evaluation and result storage
                results_categories = Dict()
                for (name, algorithm) in algorithms
                    # Evaluate rules
                    rule_evaluations = map(
                        r -> SoleModels.evaluaterule(r, logiset, y_pred, compute_explanations=true),
                        SoleModels.rules(algorithm)
                    )

                    # Store sensitivity and specificity
                    println(output_file, "ses,spec $(lowercase(name)):" *
                                         string(map(x -> (x.sensitivity, x.specificity), rule_evaluations))
                    )
                    println("ses,spec $(lowercase(name)):" *  
                            string(map(x -> (x.sensitivity, x.specificity), rule_evaluations))
                    )

                    results_categories["rule_evals_$(name)"] = rule_evaluations
                end

                # Number of terms per algorithm and detailed metrics
                println(output_file, "Number of terms:")
                println("Number of terms:")  

                for (name, algorithm) in algorithms
                    num_terms = nterm(rules(algorithm))
                    println(output_file, "$(name): $num_terms")
                    println("$(name): $num_terms")  

                    # Get the rule evaluations for this algorithm
                    rule_evaluations = results_categories["rule_evals_$(name)"]

                    # For each rule in this algorithm
                    for (rule_id, eval) in enumerate(rule_evaluations)
                        # Extract sensitivity and specificity
                        sensitivity = eval.sensitivity
                        specificity = eval.specificity

                        # Calculate explanation statistics if available
                        min_avg = 0.0
                        min_std = 0.0
                        max_avg = 0.0
                        avg_avg = 0.0
                        std_avg = 0.0

                        # Extract explanation statistics
                        natoms_expl = map(expls -> let
                                x = Float64[natoms(e) for e in expls]
                                length(unique(x)) == 1 ?
                                (minimum(x), maximum(x), mean(x), 0.0) :
                                (minimum(x), maximum(x), mean(x), std(x))
                            end,
                            filter(!isempty, eval.explanations)
                        )

                        if !isempty(natoms_expl)
                            min_avg = StatsBase.mean(Float64[x[1] for x in natoms_expl])
                            min_std = StatsBase.std(Float64[x[1] for x in natoms_expl])
                            max_avg = StatsBase.mean(Float64[x[2] for x in natoms_expl])
                            avg_avg = StatsBase.mean(Float64[x[3] for x in natoms_expl])
                            std_avg = StatsBase.mean(Float64[x[4] for x in natoms_expl])
                        end

                        # Get execution time
                        exec_time = execution_times[name]

                        # Calculate number of atoms in the rule's antecedent
                        rule = SoleModels.rules(algorithm)[rule_id]
                        rule_idd = strip(replace(string(consequent(SoleModels.rules(algorithm)[rule_id])), r"\x1B\[[0-9;]*[a-zA-Z]|▣|\n" => ""))
                        num_atoms = natoms(antecedent(rule))

                        # Write all metrics to the CSV file for this rule
                        println(csv_file, "$dataset,$name,$num_terms,$exec_time,$rule_idd,$sensitivity,$specificity,$min_avg,$min_std,$max_avg,$avg_avg,$std_avg,$num_atoms")
                    end

                    # If there are no rules (empty rule_evaluations), write a summary line with just algorithm info
                    if isempty(rule_evaluations)
                        exec_time = execution_times[name]
                        println(csv_file, "$dataset,$name,$num_terms,$exec_time,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
                    end
                end

                # Detailed rule analysis
                for (category_name, evaluations) in results_categories
                    println(output_file, "\n=== $category_name's Rules ===")
                    println(output_file, "="^40)

                    for (i, eval) in enumerate(evaluations)
                        println(output_file, "\nRule $i:")

                        # Explanation statistics
                        natoms_expl = map(expls -> let
                                x = Float64[natoms(e) for e in expls]
                                length(unique(x)) == 1 ?
                                (minimum(x), maximum(x), mean(x), 0.0) :
                                (minimum(x), maximum(x), mean(x), std(x))
                            end,
                            filter(!isempty, eval.explanations)
                        )

                        if !isempty(natoms_expl)
                            min_avg = StatsBase.mean(Float64[x[1] for x in natoms_expl])
                            min_std = StatsBase.std(Float64[x[1] for x in natoms_expl])
                            max_avg = StatsBase.mean(Float64[x[2] for x in natoms_expl])
                            avg_avg = StatsBase.mean(Float64[x[3] for x in natoms_expl])
                            std_avg = StatsBase.mean(Float64[x[4] for x in natoms_expl])

                            println(output_file, "▸ Explanation Statistics:")
                            println(output_file, "  - Min-Avg:      $(round(min_avg, digits=5))")
                            println(output_file, "  - Min-Std:      $(round(min_std, digits=5))")
                            println(output_file, "  - Max_avg:      $(round(max_avg, digits=5))")
                            println(output_file, "  - Average:      $(round(avg_avg, digits=5))")
                            println(output_file, "  - Std Dev:      $(round(std_avg, digits=5))")
                        else
                            println(output_file, "▸ No explanations available for statistics")
                        end
                        println(output_file, "-"^40)
                    end
                end
            end
        end
    end
end