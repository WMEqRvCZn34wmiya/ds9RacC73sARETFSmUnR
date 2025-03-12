# From Forests to Minimal Formulas: A Logic-Driven Framework to Explain Ensemble Models

## Overview
This repository contains the experimental implementation of a novel framework for analyzing decision tree models across various datasets using different parameters and configurations. The experiment is structured in three primary phases that generate distinct result tables.

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
├── evaluation_results_experiment_for_plot.csv        # Will be generated
├── evaluation_results_experiment1.csv                # Will be generated
├── evaluation_results_experiment2.csv                # Will be generated
├── output_*.txt                                      # Will be generated
└── other ...
```

## Usage

```bash
# Create a virtual environment
bash> python -m venv venv
# Install Python dependencies in the virtual environment
bash> venv/bin/pip install -r requirements.txt
# Start Julia after setting the PYTHON environment variable to the Python executable of the virtual environment
bash> PYTHON=venv/bin/python julia --project=.
# Import Julia's package manager
julia> import Pkg                     
# Install all dependencies specified in the Project.toml and Manifest.toml files
julia> Pkg.instantiate()             
# Rebuild the PyCall package to use the Python from the virtual environment 
julia> Pkg.build("PyCall")           
# Run your experiment script by replacing with your actual file path
julia> include("... insert here your experiment ...")  
```

## Supported Datasets
The framework supports the following datasets:
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

### Experimenter 1: First Analysis Table Generation
1. Execute `experiment1.jl` to generate the first analysis table:
   
   ```bash
   julia> include("src/experiment1.jl")
   ```
   This script performs the following operations:
   - Trains a random forest model (comprising 10 trees) on the specified datasets ["monks-1", "monks-2", "monks-3", "hayes-roth", "balance-scale"]
   - Executes multiple algorithms [Lumen, BAtrees, Refne, Trepan, Intrees, RuleCosi+] with predefined parameters
   - Evaluates rule performance using established metrics
   - Generates comprehensive statistics in `evaluation_results_experiment1.csv`

### Experimenter 2: Second Analysis Table Generation
1. Execute `experiment2.jl` to generate the second analysis table:
   
   ```bash
   julia> include("src/experiment2.jl")
   ```
   This script conducts:
   - Training of a random forest model (with optimized number of trees) on the specified datasets ["car", "post-operative", "urinary-d1", "urinary-d2", "iris"]
   - Implementation of multiple algorithms [Lumen, BAtrees, Refne, Trepan, Intrees, RuleCosi+] with predefined parameters
   - Comprehensive rule performance evaluation
   - Detailed statistical output in `evaluation_results_experiment2.csv`

### Experimenter 3: CSV files generated for comparative analysis plots, including variations of parameter c.
1. Execute `experimenter3Plot.jl` to generate CSV files with data for different values of parameter `c`:
   ```bash
   julia> include("src/experimenter3Plot.jl")
   ```
   This script systematically:
   - Trains random forest models (each with 10 trees) on all supported datasets
   - Executes multiple algorithms for each value of `c` ranging from 0.20 to 1.0 with increments of 0.20 [Lumen, BAtrees, Refne, Trepan, Intrees, RuleCosi+]
   - Conducts thorough rule performance evaluation
   - Outputs detailed analytical statistics in `evaluation_results_experiment_for_plot_c=$c.csv`

## Key Features
- Comprehensive evaluation of decision tree models
- Robust support for multiple datasets with automated preprocessing
- Detailed parameter sensitivity analysis
- Sophisticated rule generation and evaluation
- Precise performance metrics calculation

## Technical Details

### Model Parameters
- Number of trees: 3, 5, 7, 10
- Parameter range (for `c` parameter): 0.2 to 1.0 (increments of 0.2)
- Evaluation metrics include:
  - Model accuracy
  - Rule sensitivity and specificity
  - Explanation statistical measures

### Output Metrics
The experiment generates detailed metrics including:
- Forest model accuracy
- Comprehensive rule evaluation statistics
- Explanation complexity measures
- Advanced pattern analysis results

## Requirements
- Julia programming environment (version 1.10.x)
- Python environment for RuleCosi+ algorithm