# From Forests to Minimal Formulas: A Logic-Driven Framework to Explain Ensemble Models

## Overview
This repository contains the experimental of new framework, for analyzing decision tree models across different datasets using various parameters and configurations. The experiment consists of two main phases that generate separate result tables, followed by post-processing and visualization steps.

## Project Structure
```
.
├── src/
│   ├── experiment1.jl
│   ├── experiment2.jl
│   ├── experimenter3Plot.jl
│   ├── suitfortest.jl
│   ├── utilsForTest.jl
│   └── datasetDirectory and more...
├── evaluation_results_experiment_for_plot.csv        ! will be created. 
├── evaluation_results_experiment1.csv                ! will be created.
├── evaluation_results_experiment2.csv                ! will be created.
├── output_*.txt                                      ! will be created.
└── other ...
```

## Usage

```bash
# Crea un virtual environment 
 bash> python -m venv venv

# Installa le dipendenze di python nell'ambiente virtuale
 bash> venv/bin/pip install -r requirements.txt

# Avvia julia dopo aver valorizzato la variabile d'ambiente PYTHON all'eseguibile python dell'ambiente virtuale
 bash> PYTHON=venv/bin/python julia --project=.
julia> import Pkg
julia> Pkg.instantiate()
julia> Pkg.build("PyCall")
julia>  include("... insert here your experiment ...")
```

## Supported Datasets
The experiment supports the following datasets:
- monks-1
- monks-2
- monks-3
- hayes-roth
- balance-scale
- car 
- post-operative
- urinary-d1
- urinary-d2
- iris

## Experiment Execution

### Phase 1: First Table Generation
1. Run `experiment1.jl` to generate the first analysis table:
   
   ```bash
   julia> include("src/experiment1.jl")
   ```
   This script:
   - Trains a random forest model (with 10 trees) on the specified dataset ["monks-1", "monks-2", "monks-3", "hayes-roth", "balance-scale"]
   - Runs the different algorithm [Lumen, BAtrees, Refne, Trepan, Intrees, RuleCosi+] with specified parameters
   - Evaluates rule performance
   - Outputs detailed statistics in evaluation_results_experiment1.csv
### Phase 2: Second Table Generation
1. Run `experiment2.jl` to generate the second analysis table:
   
   ```bash
   julia> include("src/experiment2.jl")
   ```
   This script:
   - Trains a random forest model (with best number of trees) on the specified dataset ["car", "post-operative", "urinary-d1", "urinary-d2", "iris"]
   - Runs the different algorithm [Lumen, BAtrees, Refne, Trepan, Intrees, RuleCosi+] with specified parameters
   - Evaluates rule performance
   - Outputs detailed statistics in evaluation_results_experiment2.csv

### Phase 3: Plot Generation
1. Run `experimenter3Plot.jl` varius plot for analysis and study about many/different values of `c`:
   ```bash
   julia> include("src/experimenter3Plot.jl")
   ```
   This script:
   - Trains a random forest model ((with 10 trees)) on the specified dataset [all upported Datasets]
 - Runs the different algorithm (for each value of c between 0.20 and 1.0 with a step of 0.20) [Lumen, BAtrees, Refne, Trepan, Intrees, RuleCosi+] with specified parameters
   - Evaluates rule performance
   - Outputs detailed statistics in evaluation_results_experiment_for_plot_c=$c.csv


## Data Analysis and Visualization
The repository includes Python scripts for visualizing the experimental results. These scripts can be used to:
- Plot performance metrics
- Compare results across different datasets
- Analyze the impact of different parameter combinations
- Visualize rule evaluation statistics

## Key Features
- Comprehensive evaluation of decision tree models
- Support for multiple datasets with automatic preprocessing
- Parameter sensitivity analysis
- Rule generation and evaluation
- Performance metrics calculation

## Technical Details

### Model Parameters
- Number of trees: 3, 5, 7, 10
- Parameter range: 0.2 to 1.0 (increments of 0.2)
- Evaluation metrics include:
  - Model accuracy
  - Rule sensitivity and specificity
  - Explanation statistics

### Output Metrics
The experiment generates detailed metrics including:
- Forest model accuracy
- Rule evaluation statistics
- Explanation complexity measures
- Pattern analysis results

## Requirements
- Julia programming environment
- Required Julia packages:
- DecisionTree
- DataFrames
- StatsBase
- SoleModels
- SoleData
- SoleLogics
- SolePostHoc #Notes (the core algorithm of the experiment) is located in the `lumen-dirty` branch.
- Python for RuleCosi+ algorithm
- Python with plotting libraries (for visualization)
