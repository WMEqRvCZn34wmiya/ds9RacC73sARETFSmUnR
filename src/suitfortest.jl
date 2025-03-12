function nterm(rule)

    antecedent = rule.antecedent
    

    function count_or_recursive(node)

        if !(typeof(node) <: SyntaxBranch)
            return 0
        end
        

        is_or = (typeof(node.token) <: NamedConnective{:∨}) ? 1 : 0
        
        count_in_children = 0
        if isdefined(node, :children)
            for child in node.children
                count_in_children += count_or_recursive(child)
            end
        end
        
        return is_or + count_in_children
    end
    

    return count_or_recursive(antecedent) + 1
end

function nterm(rules::Vector)
    return map(nterm, rules)
end

function calcola_percentuali(vettore::Vector{String})
    total = length(vettore)
    
    count = Dict{String, Int}()
    
    for string in vettore
        count[string] = get(count, string, 0) + 1
    end
    
    percentages = Dict{String, Float64}()
    for (string, cnt) in count
        percentages[string] = (cnt / total) * 100
    end
    
    return percentages
end

function analizza_bilanciamento_su_file(labels_train::Vector{String}, labels_test::Vector{String}, file_name::String="dataset_balance_analysis.txt")

    open(file_name, "w") do file

        train_total = length(labels_train)
        train_percentages = calcola_percentuali(labels_train)
        train_count = Dict{String, Int}()
        
        for string in labels_train
            train_count[string] = get(train_count, string, 0) + 1
        end
        
        write(file, "DATASET BALANCE ANALYSIS\n")
        write(file, "==========================\n\n")
        write(file, "Train dataset:\n")
        write(file, "Total elements: $train_total\n")
        write(file, "Percentages: $train_percentages\n\n")
        write(file, "Train count details:\n")
        
        for (string, cnt) in train_count
            percentage = (cnt / train_total) * 100
            write(file, "'$string': $cnt occurrences ($(round(percentage, digits=2))%)\n")
        end
        
        # Test dataset analysis
        test_total = length(labels_test)
        test_percentages = calcola_percentuali(labels_test)
        test_count = Dict{String, Int}()
        
        for string in labels_test
            test_count[string] = get(test_count, string, 0) + 1
        end
        
        write(file, "\nTest dataset:\n")
        write(file, "Total elements: $test_total\n")
        write(file, "Percentages: $test_percentages\n\n")
        write(file, "Test count details:\n")
        
        for (string, cnt) in test_count
            percentage = (cnt / test_total) * 100
            write(file, "'$string': $cnt occurrences ($(round(percentage, digits=2))%)\n")
        end
        
        # Train and test comparison
        write(file, "\nTRAIN/TEST COMPARISON:\n")
        write(file, "====================\n")
        unique_classes = unique(vcat(collect(keys(train_count)), collect(keys(test_count))))
        
        write(file, "Class\tTrain%\tTest%\tDiff\n")
        for class in unique_classes
            train_perc = get(train_percentages, class, 0.0)
            test_perc = get(test_percentages, class, 0.0)
            difference = abs(train_perc - test_perc)
            write(file, "$class\t$(round(train_perc, digits=2))%\t$(round(test_perc, digits=2))%\t$(round(difference, digits=2))%\n")
        end
    end
    
    println("Balance analysis saved in file '$file_name'")
end


function calcola_percentuali2(vettore::Vector{String})
    """
    Calculate the percentage of occurrence of each unique string in the vector.
    
    Args:
        vettore: A vector of strings
        
    Returns:
        A dictionary with strings as keys and their percentages as values
    """
    # Calculate the total number of elements
    total = length(vettore)
    
    # Initialize a dictionary to count occurrences
    count = Dict{String, Int}()
    
    # Count the occurrences of each string
    for string in vettore
        count[string] = get(count, string, 0) + 1
    end
    
    # Calculate the percentages
    percentages = Dict{String, Float64}()
    for (string, cnt) in count
        percentages[string] = (cnt / total) * 100
    end
    
    return percentages
end


function analizza_bilanciamento2(vettore::Vector{String})
    total = length(vettore)
    count = Dict{String, Int}()
    
    for string in vettore
        count[string] = get(count, string, 0) + 1
    end
    
    println("Total elements: $total")
    println("\nCount per class:")
    for (string, cnt) in count
        percentage = (cnt / total) * 100
        println("'$string': $cnt occurrences ($(round(percentage, digits=2))%)")
    end
    
    # Analyze the balance
    if length(count) == 1
        println("\nThe dataset is not balanced: it contains only one class.")
    elseif length(count) > 1
        min_perc = minimum(values(count)) / total * 100
        max_perc = maximum(values(count)) / total * 100
        
        if max_perc / min_perc > 3
            println("\nThe dataset is highly imbalanced (ratio > 3:1).")
        elseif max_perc / min_perc > 1.5
            println("\nThe dataset is moderately imbalanced.")
        else
            println("\nThe dataset is relatively balanced.")
        end
    end
end



function create_importance_vector(importances)
    indexed_importances = [(i, imp) for (i, imp) in enumerate(importances)]
    
    sort!(indexed_importances, by=x -> x[2], rev=true)
    
    return [idx for (idx, _) in indexed_importances]
end


