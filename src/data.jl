function load_data_hardcoded(hard_coded_dataset)
    dataset_path = "src/$hard_coded_dataset/$hard_coded_dataset.data"
    Random.seed!(2025)

    # Specific preprocessing for each dataset
    if hard_coded_dataset == "zoo"
        # Zoo: 18 columns (1 animal name, 16 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:17])  # exclude animal name and take only features
        labels = string.(data[:, 18])     # last column is the class

    elseif hard_coded_dataset == "bean"
        # bean: 
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 3:15])  # first 9 columns are features
        labels = string.(data[:, 2])     # last column is the class

    elseif hard_coded_dataset == "urinary-d1" || hard_coded_dataset == "urinary-d2"
        # Define the dataset path
        dataset_path = "src/diagnosis/diagnosis.data"
        
        # Read the file as binary to remove any BOM
        raw_data = read(dataset_path)
        
        # Convert binary data to string
        content = String(raw_data)
        
        # Split by lines
        lines = split(content, '\n')
        
        # Remove empty lines
        lines = filter(line -> !isempty(strip(line)), lines)
        
        # Initialize the array for data
        data = []
        
        for line in lines
            try
                # Split by comma
                parts = split(line, ',')
                
                # Ensure there are at least 8 columns
                if length(parts) < 8
                    println("Skipped line, less than 8 columns: ", line)
                    continue
                end
                
                # Construct a data row
                row = []
                
                # Convert temperature (first column) to Float64
                # Ensure to replace comma with dot if necessary
                temp_str = replace(parts[1], ',' => '.')
                push!(row, parse(Float64, temp_str))
                
                # Add other columns as strings
                for i in 2:length(parts)
                    push!(row, strip(parts[i]))
                end
                
                push!(data, row)
            catch e
                println("Error processing line: ", line)
                println("Error: ", e)
                continue
            end
        end
        
        # Create an empty array for numerical features
        num_rows = length(data)
        numerical_features = zeros(Float64, num_rows, 6)
        
        # Process the features
        for i in 1:num_rows
            # Feature 1: Patient's temperature (already Float64)
            numerical_features[i, 1] = data[i][1]
            
            # Features 2-6: Binary yes/no values
            for j in 2:6
                feature_value = data[i][j]
                # Convert "yes" to 1.0 and "no" to 0.0
                numerical_features[i, j] = lowercase(strip(feature_value)) == "yes" ? 1.0 : 0.0
            end
        end
        
        # Set the features
        features = numerical_features
        
        # Set the labels based on the dataset type
        labels = String[]
        
        # If urinary-d1, use column 7, if urinary-d2 use column 8
        label_column = hard_coded_dataset == "urinary-d1" ? 7 : 8
        
        for i in 1:num_rows
            if label_column <= length(data[i])
                label_value = data[i][label_column]
                # Convert to "yes" or "no"
                push!(labels, lowercase(strip(label_value)) == "yes" ? "yes" : "no")
            else
                println("Warning: row $i does not have enough columns for the label")
                # Set a default value
                push!(labels, "no")
            end
        end
        
    
    elseif hard_coded_dataset == "post-operative"
        Random.seed!(1004)

        data = DelimitedFiles.readdlm(dataset_path, ',')
        
        # Filter out rows containing "?" values
        valid_rows = []
        for i in 1:size(data, 1)
            if !any(x -> x == "?", data[i, :])
                push!(valid_rows, i)
            end
        end
        
        # Subset the data to only include valid rows
        filtered_data = data[valid_rows, :]
        
        # Create empty array to hold transformed numerical features
        num_rows = size(filtered_data, 1)
        numerical_features = zeros(Float64, num_rows, 8)
        
        # Feature 1: L-CORE (patient's internal temperature)
        for i in 1:num_rows
            if filtered_data[i, 1] == "high"
                numerical_features[i, 1] = 2.0
            elseif filtered_data[i, 1] == "mid"
                numerical_features[i, 1] = 1.0
            elseif filtered_data[i, 1] == "low"
                numerical_features[i, 1] = 0.0
            end
        end
        
        # Feature 2: L-SURF (patient's surface temperature)
        for i in 1:num_rows
            if filtered_data[i, 2] == "high"
                numerical_features[i, 2] = 2.0
            elseif filtered_data[i, 2] == "mid"
                numerical_features[i, 2] = 1.0
            elseif filtered_data[i, 2] == "low"
                numerical_features[i, 2] = 0.0
            end
        end
        
        # Feature 3: L-O2 (oxygen saturation)
        for i in 1:num_rows
            if filtered_data[i, 3] == "excellent"
                numerical_features[i, 3] = 3.0
            elseif filtered_data[i, 3] == "good"
                numerical_features[i, 3] = 2.0
            elseif filtered_data[i, 3] == "fair"
                numerical_features[i, 3] = 1.0
            elseif filtered_data[i, 3] == "poor"
                numerical_features[i, 3] = 0.0
            end
        end
        
        # Feature 4: L-BP (blood pressure)
        for i in 1:num_rows
            if filtered_data[i, 4] == "high"
                numerical_features[i, 4] = 2.0
            elseif filtered_data[i, 4] == "mid"
                numerical_features[i, 4] = 1.0
            elseif filtered_data[i, 4] == "low"
                numerical_features[i, 4] = 0.0
            end
        end
        
        # Feature 5: SURF-STBL (stability of surface temperature)
        for i in 1:num_rows
            if filtered_data[i, 5] == "stable"
                numerical_features[i, 5] = 2.0
            elseif filtered_data[i, 5] == "mod-stable"
                numerical_features[i, 5] = 1.0
            elseif filtered_data[i, 5] == "unstable"
                numerical_features[i, 5] = 0.0
            end
        end
        
        # Feature 6: CORE-STBL (stability of core temperature)
        for i in 1:num_rows
            if filtered_data[i, 6] == "stable"
                numerical_features[i, 6] = 2.0
            elseif filtered_data[i, 6] == "mod-stable"
                numerical_features[i, 6] = 1.0
            elseif filtered_data[i, 6] == "unstable"
                numerical_features[i, 6] = 0.0
            end
        end
        
        # Feature 7: BP-STBL (stability of blood pressure)
        for i in 1:num_rows
            if filtered_data[i, 7] == "stable"
                numerical_features[i, 7] = 2.0
            elseif filtered_data[i, 7] == "mod-stable"
                numerical_features[i, 7] = 1.0
            elseif filtered_data[i, 7] == "unstable"
                numerical_features[i, 7] = 0.0
            end
        end
        
        # Feature 8: COMFORT (patient's comfort at discharge)
        for i in 1:num_rows
            # Handle different types properly
            comfort_value = filtered_data[i, 8]
            if isa(comfort_value, String)
                numerical_features[i, 8] = parse(Float64, comfort_value)
            elseif isa(comfort_value, Number)
                numerical_features[i, 8] = Float64(comfort_value)
            end
        end
        
        # Set up the features and labels
        features = numerical_features
        labels = string.(filtered_data[:, 9])  # Convert to string to ensure proper type



    elseif hard_coded_dataset == "madman"
        # Heberman: 4 columns (1 class, 3 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 1:3])  
        labels = string.(data[:, 4])   


    elseif hard_coded_dataset == "BankNote_Authentication_UCI"
        # BankNote_Authentication_UCI: 10 columns (1 class, 9 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 1:4])  # first 9 columns are features
        labels = string.(data[:, 5])     # last column is the class


    elseif hard_coded_dataset == "lenses"
        # Glass: 10 columns (1 ID, 9 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:5])  # exclude ID and take only features
        labels = string.(data[:, 1])     # last column is the class

    elseif hard_coded_dataset == "hepatitis"
        # Hepatitis: 20 columns (1 class, 19 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')

        data_processed = Matrix{Any}(data)

        for i in 1:size(data_processed, 1)
            for j in 1:size(data_processed, 2)
                if data_processed[i, j] == "?"
                    data_processed[i, j] = missing  # oppure NaN
                end
            end
        end

       
        labels = string.(data_processed[:, 1])

        features = Array{Float64}(undef, size(data_processed, 1), 19)
        for i in 1:size(data_processed, 1)
            for j in 2:20  
                if ismissing(data_processed[i, j])
                    #features[i, j-1] = 0.0 
                else
                    features[i, j-1] = Float64(data_processed[i, j])
                end
            end
        end


    elseif hard_coded_dataset == "lymphography"
        # Lymphography: 19 columns (1 class, 18 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:19])  # exclude the first column (class) and take only features
        labels = string.(data[:, 1])      # first column is the class

    elseif hard_coded_dataset == "glass"
        # Glass: 10 columns (1 ID, 9 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:10])  # exclude ID and take only features
        labels = string.(data[:, 11])     # last column is the class

    elseif hard_coded_dataset == "penguins"
        # Penguins: 7 columns (species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        sex_column = data[:, 7]  
        numeric_sex = map(sex -> sex == "FEMALE" ? 1.0 : 0.0, sex_column)
        numerical_features = float.(data[:, 3:6])  # Features originali
        features = hcat(numerical_features, numeric_sex)  
        labels = string.(data[:, 1]) 

    elseif hard_coded_dataset == "mushroom"
        data = DelimitedFiles.readdlm(dataset_path, ',')
        mappings = Dict(
           
            2 => Dict("b" => 0.0, "c" => 1.0, "x" => 2.0, "f" => 3.0, "k" => 4.0, "s" => 5.0),
            
            3 => Dict("f" => 0.0, "g" => 1.0, "y" => 2.0, "s" => 3.0),
            
            4 => Dict("n" => 0.0, "b" => 1.0, "c" => 2.0, "g" => 3.0, "r" => 4.0, "p" => 5.0, "u" => 6.0, "e" => 7.0, "w" => 8.0, "y" => 9.0),
            
            5 => Dict("t" => 1.0, "f" => 0.0),
           
        )

        num_rows = size(data, 1)
        num_cols = size(data, 2) - 1  
        numeric_features = zeros(Float64, num_rows, num_cols)

        for col in 2:size(data, 2)  
            feature_idx = col - 1  

            if haskey(mappings, col)
                
                for row in 1:num_rows
                    val = data[row, col]
                    numeric_features[row, feature_idx] = mappings[col][val]
                end
            else
               
                unique_vals = unique(data[:, col])
                col_mapping = Dict(val => Float64(i - 1) for (i, val) in enumerate(unique_vals))

                for row in 1:num_rows
                    val = data[row, col]
                    numeric_features[row, feature_idx] = col_mapping[val]
                end
            end
        end


        features = numeric_features

        labels = map(label -> label == "e" ? "edible" : "poisonous", data[:, 1])

    elseif hard_coded_dataset == "heart"
        # Heart: 14 columns (1 class, 13 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 1:13])  # first 13 columns are features
        labels = string.(data[:, 14])     # last column is the class


    elseif hard_coded_dataset == "monks-1"
        Random.seed!(1010)
        dataset_path = "src/monks/$hard_coded_dataset.data"
        # Monks: 7 columns (1 class, 6 features, 1 extra to ignore)
        data = DelimitedFiles.readdlm(dataset_path, ' ', skipblanks=true)
        features = float.(data[:, 2:7])   # from the second to the seventh column are features
        labels = string.(data[:, 1])      # first column is the class
        println("features: $features")
        println("labels: $labels")

    elseif hard_coded_dataset == "monks-2"
        Random.seed!(101010)
        dataset_path = "src/monks/$hard_coded_dataset.data"
        # Monks: 7 columns (1 class, 6 features, 1 extra to ignore)
        data = DelimitedFiles.readdlm(dataset_path, ' ', skipblanks=true)
        features = float.(data[:, 2:7])   # from the second to the seventh column are features
        labels = string.(data[:, 1])      # first column is the class
        println("features: $features")
        println("labels: $labels")

    elseif hard_coded_dataset == "monks-3"
        Random.seed!(10101010)
        dataset_path = "src/monks/$hard_coded_dataset.data"
        # Monks: 7 columns (1 class, 6 features, 1 extra to ignore)
        data = DelimitedFiles.readdlm(dataset_path, ' ', skipblanks=true)
        features = float.(data[:, 2:7])   # from the second to the seventh column are features
        labels = string.(data[:, 1])      # first column is the class
        println("features: $features")
        println("labels: $labels")

    elseif hard_coded_dataset == "tae"
        # TAE: 6 columns (5 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')

        # All columns are already numeric except for the class
        features = float.(data[:, 1:5])  # first 5 columns are features

        # Modify the fifth feature based on ranges

        for i in 1:size(features, 1)
            if features[i, 5] >= 0 && features[i, 5] <= 20
                features[i, 5] = 0
            elseif features[i, 5] > 20 && features[i, 5] <= 40
                features[i, 5] = 1
            else  # for values > 40
                features[i, 5] = 2
            end
        end

        labels = string.(data[:, 6])     # last column is the class (1,2,3)

    elseif hard_coded_dataset == "car"
        Random.seed!(10)
        # Car: 7 columns (6 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 6)

        # Map for buying and maint
        price_map = Dict("vhigh" => 4.0, "high" => 3.0, "med" => 2.0, "low" => 1.0)

        # Map for doors
        doors_map = Dict("2" => 2.0, "3" => 3.0, "4" => 4.0, "5more" => 5.0)

        # Map for persons
        persons_map = Dict("2" => 2.0, "4" => 4.0, "more" => 6.0)

        # Map for lug_boot
        boot_map = Dict("small" => 1.0, "med" => 2.0, "big" => 3.0)

        # Map for safety
        safety_map = Dict("low" => 1.0, "med" => 2.0, "high" => 3.0)

        for i in 1:size(data, 1)
            # buying (column 1)
            features[i, 1] = price_map[string(data[i, 1])]

            # maint (column 2)
            features[i, 2] = price_map[string(data[i, 2])]

            # doors (column 3)
            features[i, 3] = doors_map[string(data[i, 3])]

            # persons (column 4)
            features[i, 4] = persons_map[string(data[i, 4])]

            # lug_boot (column 5)
            features[i, 5] = boot_map[string(data[i, 5])]

            # safety (column 6)
            features[i, 6] = safety_map[string(data[i, 6])]
        end
        labels = string.(data[:, 7])  # last column is the class

    elseif hard_coded_dataset == "tictactoe"
        # Tictactoe: 10 columns (9 features for the grid, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 9)
        for i in 1:size(data, 1)
            for j in 1:9  # first 9 columns are the grid positions
                if data[i, j] == "x"
                    features[i, j] = 1.0   # x -> 1.0
                elseif data[i, j] == "o"
                    features[i, j] = 0.0   # o -> 0.0
                else
                    features[i, j] = -1.0  # in case of missing value or other
                end
            end
        end
        labels = string.(data[:, 10])  # last column is the class


    elseif hard_coded_dataset == "house-votes"
        # House-votes: 17 columns (1 class, 16 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        # Convert y/n/? to 1/0/2
        features = zeros(Float64, size(data, 1), 16)
        for i in 1:size(data, 1)
            for j in 2:17  # skip the first column (class)
                if data[i, j] == "y"
                    features[i, j-1] = 1.0
                elseif data[i, j] == "n"
                    features[i, j-1] = 0.0
                else  # case "?"
                    features[i, j-1] = 2.0
                end
            end
        end
        labels = string.(data[:, 1])  # first column is the class

    elseif hard_coded_dataset == "iris"
        features, labels = load_data(hard_coded_dataset)
        features = float.(features)
        labels = string.(labels)

    elseif hard_coded_dataset == "hayes-roth"
        # Hayes-roth: 6 columns (1 to ignore, 4 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:5])  # from the second to the fifth column are features
        labels = string.(data[:, 6])     # last column is the class

    elseif hard_coded_dataset == "primary-tumor"
        # Primary-tumor: 18 columns (1 class, 17 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 17)
        for i in 1:size(data, 1)
            for j in 2:18  # skip the first column (class)
                if data[i, j] == "?"
                    features[i, j-1] = -1.0  # replace ? with -1
                else
                    features[i, j-1] = Float64(data[i, j])  # convert directly to Float64
                end
            end
        end
        labels = string.(data[:, 1])  # first column is the class

    elseif hard_coded_dataset == "soybean-small"
        # Soybean-small: 36 columns (35 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 1:35])  # first 35 columns are features
        labels = string.(data[:, 36])     # last column is the class

    elseif hard_coded_dataset == "cmc"
        # CMC: 10 columns (9 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')

        # Handling features: the first 9 columns
        features = float.(data[:, 1:9])

        # Modify wife age (first column) into ranges
        for i in 1:size(features, 1)
            if features[i, 1] <= 18
                features[i, 1] = 0        # up to 18 years
            elseif features[i, 1] <= 30
                features[i, 1] = 1        # from 19 to 30 years
            elseif features[i, 1] <= 35
                features[i, 1] = 2        # from 31 to 35 years
            else
                features[i, 1] = 3        # over 35 years
            end
        end

        # Columns 2,3,7,8 are already categorical (1,2,3,4)
        # Columns 5,6,9 are already binary (0,1)
        # Column 4 is numeric and we leave it as is

        # The last column (10) is the class
        labels = string.(data[:, 10])  # classes: 1=No-use, 2=Long-term, 3=Short-term

    elseif hard_coded_dataset == "balance-scale"
        # Balance-scale: 5 columns (1 class, 4 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:5])  # from the second to the fifth column are features
        labels = string.(data[:, 1])     # first column is the class
    end


    # Split train-test (80-20)
    n = size(features, 1)
    indices = Random.shuffle(1:n)
    train_size = floor(Int, 0.8 * n)

    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]

    features_train = features[train_indices, :]
    labels_train = labels[train_indices]
    features_test = features[test_indices, :]
    labels_test = labels[test_indices]

    # Save training and test data
    mkpath("src/owncsv")
    DelimitedFiles.writedlm("src/owncsv/X_train.csv", features_train, ',')
    DelimitedFiles.writedlm("src/owncsv/y_train.csv", labels_train, ',')
    DelimitedFiles.writedlm("src/owncsv/X_test.csv", features_test, ',')
    DelimitedFiles.writedlm("src/owncsv/y_test.csv", labels_test, ',')

    #features = features_train  # use the training set for the model
    #labels = labels_train

    return features, labels, features_train, labels_train, features_test, labels_test
end

"""
    load_data_hc(hard_coded_dataset::String)

Load and preprocess data for a specified hard-coded dataset.

# Arguments
- `hard_coded_dataset::String`: The name of the dataset to load. Must be one of the supported datasets.

# Supported Datasets
- "iris"
- "zoo"
- "monks"
- "house-votes"
- "balance-scale"
- "hayes-roth"
- "primary-tumor"
- "soybean-small"
- "tictactoe"
- "car"
- "tae"
- "cmc"

# Description
This function loads and preprocesses data for the specified hard-coded dataset. Each dataset has specific preprocessing steps to handle its unique structure and features. The function returns the features and labels of the dataset.

# Output
- A tuple `(features, labels)` where `features` is a matrix of feature values and `labels` is a vector of class labels.

# Example
```julia
features, labels = load_data_hc("iris")
"""
function load_data_hc(hard_coded_dataset)
    if !(hard_coded_dataset in ["iris", "zoo", "monks", "house-votes", "balance-scale", "hayes-roth", "primary-tumor", "soybean-small", "tictactoe", "car", "tae", "heart"])
        error("Dataset $hard_coded_dataset not supported")
    end

    dataset_path = "src/$hard_coded_dataset/$hard_coded_dataset.data"

    if hard_coded_dataset == "zoo"
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:17])
        labels = string.(data[:, 18])

    elseif hard_coded_dataset == "heart"
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 1:13])
        labels = string.(data[:, 14])

    elseif hard_coded_dataset == "iris"
        features, labels = load_data(hard_coded_dataset)
        features = float.(features)
        labels = string.(labels)
        features, labels

    elseif hard_coded_dataset == "monks"
        data = DelimitedFiles.readdlm(dataset_path, ' ', skipblanks=true)
        features = float.(data[:, 2:7])
        labels = string.(data[:, 1])

    elseif hard_coded_dataset == "tictactoe"
        # Tictactoe: 10 columns (9 features for the grid, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 9)
        for i in 1:size(data, 1)
            for j in 1:9  # first 9 columns are the grid positions
                if data[i, j] == "x"
                    features[i, j] = 1.0   # x -> 1.0
                elseif data[i, j] == "o"
                    features[i, j] = 0.0   # o -> 0.0
                else
                    features[i, j] = -1.0  # in case of missing value or other
                end
            end
        end
        labels = string.(data[:, 10])  # last column is the class


    elseif hard_coded_dataset == "house-votes"
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 16)
        for i in 1:size(data, 1)
            for j in 2:17
                if data[i, j] == "y"
                    features[i, j-1] = 1.0
                elseif data[i, j] == "n"
                    features[i, j-1] = 0.0
                else
                    features[i, j-1] = 2.0
                end
            end
        end
        labels = string.(data[:, 1])

    elseif hard_coded_dataset == "hayes-roth"
        # Hayes-roth: 6 columns (1 to ignore, 4 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:5])  # from the second to the fifth column are features
        labels = string.(data[:, 6])     # last column is the class

    elseif hard_coded_dataset == "primary-tumor"
        # Primary-tumor: 18 columns (1 class, 17 features)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 17)
        for i in 1:size(data, 1)
            for j in 2:18  # skip the first column (class)
                if data[i, j] == "?"
                    features[i, j-1] = -1.0  # replace ? with -1
                else
                    features[i, j-1] = Float64(data[i, j])  # convert directly to Float64
                end
            end
        end
        labels = string.(data[:, 1])  # first column is the class

    elseif hard_coded_dataset == "soybean-small"
        # Soybean-small: 36 columns (35 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 1:35])  # first 35 columns are features
        labels = string.(data[:, 36])     # last column is the class

    elseif hard_coded_dataset == "tae"
        # TAE: 6 columns (5 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')

        # All columns are already numeric except for the class
        features = float.(data[:, 1:5])  # first 5 columns are features

        # Modify the fifth feature based on ranges
        for i in 1:size(features, 1)
            if features[i, 5] >= 0 && features[i, 5] <= 20
                features[i, 5] = 0
            elseif features[i, 5] > 20 && features[i, 5] <= 40
                features[i, 5] = 1
            else  # for values > 40
                features[i, 5] = 2
            end
        end

        labels = string.(data[:, 6])     # last column is the class (1,2,3)

    elseif hard_coded_dataset == "car"
        # Car: 7 columns (6 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = zeros(Float64, size(data, 1), 6)

        # Map for buying and maint
        price_map = Dict("vhigh" => 4.0, "high" => 3.0, "med" => 2.0, "low" => 1.0)

        # Map for doors
        doors_map = Dict("2" => 2.0, "3" => 3.0, "4" => 4.0, "5more" => 5.0)

        # Map for persons
        persons_map = Dict("2" => 2.0, "4" => 4.0, "more" => 6.0)

        # Map for lug_boot
        boot_map = Dict("small" => 1.0, "med" => 2.0, "big" => 3.0)

        # Map for safety
        safety_map = Dict("low" => 1.0, "med" => 2.0, "high" => 3.0)

        for i in 1:size(data, 1)
            # buying (column 1)
            features[i, 1] = price_map[string(data[i, 1])]

            # maint (column 2)
            features[i, 2] = price_map[string(data[i, 2])]

            # doors (column 3)
            features[i, 3] = doors_map[string(data[i, 3])]

            # persons (column 4)
            features[i, 4] = persons_map[string(data[i, 4])]

            # lug_boot (column 5)
            features[i, 5] = boot_map[string(data[i, 5])]

            # safety (column 6)
            features[i, 6] = safety_map[string(data[i, 6])]
        end

        labels = string.(data[:, 7])  # last column is the class

    elseif hard_coded_dataset == "cmc"
        # CMC: 10 columns (9 features, 1 class)
        data = DelimitedFiles.readdlm(dataset_path, ',')

        # Handling features: the first 9 columns
        features = float.(data[:, 1:9])

        # Modify wife age (first column) into ranges
        for i in 1:size(features, 1)
            if features[i, 1] <= 18
                features[i, 1] = 0        # up to 18 years
            elseif features[i, 1] <= 30
                features[i, 1] = 1        # from 19 to 30 years
            elseif features[i, 1] <= 35
                features[i, 1] = 2        # from 31 to 35 years
            else
                features[i, 1] = 3        # over 35 years
            end
        end

        # Columns 2,3,7,8 are already categorical (1,2,3,4)
        # Columns 5,6,9 are already binary (0,1)
        # Column 4 is numeric and we leave it as is

        # The last column (10) is the class
        labels = string.(data[:, 10])  # classes: 1=No-use, 2=Long-term, 3=Short-term

    elseif hard_coded_dataset == "balance-scale"
        data = DelimitedFiles.readdlm(dataset_path, ',')
        features = float.(data[:, 2:5])
        labels = string.(data[:, 1])
    end

    return features, labels
end
