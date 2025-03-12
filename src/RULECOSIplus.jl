__precompile__()


using Pkg

using PyCall
using Conda
using DataFrames
using CSV
using Random
using JSON
using SoleModels     
using SoleLogics
using DataStructures

include("apiRuleCosi.jl")

if !@isdefined rulecosi
    const rulecosi = PyNULL()
end
if !@isdefined sklearn
    const sklearn = PyNULL()
end

Conda.pip_interop(true, PyCall.Conda.ROOTENV)
PyCall.Conda.pip("install", "git+https://github.com/jobregon1212/rulecosi.git", PyCall.Conda.ROOTENV)
PyCall.Conda.pip("install", "scikit-learn", PyCall.Conda.ROOTENV)

copy!(rulecosi, pyimport("rulecosi"))
copy!(sklearn, pyimport("sklearn.ensemble"))

##############################
# 1) Struttura e BFS/DFS
##############################

mutable struct SkNode
    left::Int
    right::Int
    feature::Int
    threshold::Float64
    counts::Vector{Float64}
end

"""
build_sklearnlike_arrays(branch, n_classes, class_to_idx)

Performs post-order DFS on a Branch{String} (or ConstantModel{String}),
creating children_left, children_right, feature, threshold, value, n_nodes.
"""
function build_sklearnlike_arrays(branch, n_classes::Int, class_to_idx::Dict{String,Int})
    nodes = SkNode[]
    function dfs(b)
        i = length(nodes)
        push!(nodes, SkNode(-1, -1, -2, -2.0, fill(0.0, n_classes)))
        if b isa ConstantModel{String}
            cidx = haskey(class_to_idx, b.outcome) ? class_to_idx[b.outcome] : 0
            nodes[i+1].counts[cidx+1] = 10.0
        elseif b isa Branch{String}
            thr = b.antecedent.value.threshold
            fx  = b.antecedent.value.metacond.feature.i_variable - 1
            left_i  = dfs(b.posconsequent)
            right_i = dfs(b.negconsequent)
            nodes[i+1].feature   = fx
            nodes[i+1].threshold = thr
            nodes[i+1].left      = left_i
            nodes[i+1].right     = right_i
            for c in 1:n_classes
                nodes[i+1].counts[c] = nodes[left_i+1].counts[c] + nodes[right_i+1].counts[c]
            end
        else
            error("Unknown node type: $(typeof(b))")
        end
        return i
    end
    dfs(branch)
    n_nodes = length(nodes)
    children_left  = Vector{Int}(undef, n_nodes)
    children_right = Vector{Int}(undef, n_nodes)
    feature        = Vector{Int}(undef, n_nodes)
    threshold      = Vector{Float64}(undef, n_nodes)
    value = Array{Float64,3}(undef, n_nodes, 1, n_classes)
    for i in 1:n_nodes
        children_left[i]  = nodes[i].left
        children_right[i] = nodes[i].right
        feature[i]        = nodes[i].feature
        threshold[i]      = nodes[i].threshold
        for c in 1:n_classes
            value[i,1,c] = nodes[i].counts[c]
        end
    end
    return children_left, children_right, feature, threshold, value, n_nodes
end

function serialize_branch_sklearn(b, classes::Vector{String}, class_to_idx::Dict{String,Int})
    n_classes = length(classes)
    cl, cr, ft, th, val, nn = build_sklearnlike_arrays(b, n_classes, class_to_idx)
    posfeats = ft[ft .>= 0]
    maxfeat = isempty(posfeats) ? 0 : maximum(posfeats)
    n_features = maxfeat + 1
    return Dict(
        "children_left" => cl,
        "children_right" => cr,
        "feature" => ft,
        "threshold" => th,
        "value" => val,
        "node_count" => nn,
        "n_features" => n_features,
        "n_classes" => n_classes
    )
end

"""
serialize_julia_ensemble(ensemble, classes)

Create dict:
  "classes" => classes,
  "trees"   => [<tree_state1>, <tree_state2>, ...]
"""
function serialize_julia_ensemble(ensemble, classes::Vector{String})
    class_to_idx = Dict{String,Int}()
    for (i, cl) in enumerate(classes)
        class_to_idx[cl] = i-1
    end
    trees = [serialize_branch_sklearn(b, classes, class_to_idx) for b in ensemble.models]
    return Dict("classes" => classes, "trees" => trees)
end

##############################
# 2) BLOCK PYTHON
##############################

py"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
from collections import Counter
import io, sys

def build_sklearn_tree(tree_state):
    n_nodes = tree_state["node_count"]
    n_features = tree_state["n_features"]
    n_classes = tree_state["n_classes"]
    t = Tree(n_features, np.array([n_classes], dtype=np.intp), 1)
    dt = np.dtype([
        ('left_child','<i8'),
        ('right_child','<i8'),
        ('feature','<i8'),
        ('threshold','<f8'),
        ('impurity','<f8'),
        ('n_node_samples','<i8'),
        ('weighted_n_node_samples','<f8'),
        ('missing_go_to_left','u1')
    ])
    cl = tree_state["children_left"]
    cr = tree_state["children_right"]
    ft = tree_state["feature"]
    th = tree_state["threshold"]
    v_orig = tree_state["value"]
    v = np.ascontiguousarray(v_orig, dtype=np.float64)
    nodes_list = []
    for i in range(n_nodes):
        lc = cl[i]
        rc = cr[i]
        fe = ft[i]
        tr = th[i]
        counts = v[i,0,:]
        tot = int(np.sum(counts))
        imp = 0.0
        nodes_list.append((lc, rc, fe, tr, imp, tot, float(tot), 0))
    structured_nodes = np.array(nodes_list, dtype=dt)
    st = {"max_depth": 20, "node_count": n_nodes, "nodes": structured_nodes, "values": v}
    t.__setstate__(st)
    return t

class RealDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, tree_state, classes):
        super().__init__()
        self.tree_ = build_sklearn_tree(tree_state)
        self.n_features_ = tree_state["n_features"]
        self.n_outputs_ = 1
        self.n_classes_ = tree_state["n_classes"]
        self.classes_ = np.array(classes)
        self.fitted_ = True
    def fit(self, X, y=None):
        return self

class RealRandomForestClassifier(RandomForestClassifier):
    def __init__(self, forest_json):
        super().__init__(n_estimators=0)
        self.classes_ = np.array(forest_json["classes"])
        self.estimators_ = []
        for tree_state in forest_json["trees"]:
            self.estimators_.append(RealDecisionTreeClassifier(tree_state, forest_json["classes"]))
        self.fitted_ = True
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        arr = np.array([est.predict(X) for est in self.estimators_])
        final = []
        for i in range(X.shape[0]):
            c = Counter(arr[:, i]).most_common(1)[0][0]
            final.append(c)
        return np.array(final)

def build_sklearn_model_from_julia(dict_model):
    return RealRandomForestClassifier(dict_model)

def get_simplified_rules(rc, heuristics_digits=4, condition_digits=1):
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        rc.simplified_ruleset_.print_rules(heuristics_digits=heuristics_digits, condition_digits=condition_digits)
    finally:
        sys.stdout = old_stdout
    rules_str = buffer.getvalue()
    return rules_str.strip().splitlines()
"""

##############################
# 3) FUN process_rules_decision_list
##############################
"""
process_rules_decision_list(rules::Vector{String})

Processes raw rules from RuleCOSI into a vector of tuples (condition, outcome).
The last rule with empty condition is treated as the default rule.
"""
function process_rules_decision_list(rules::Vector{String})
    structured = Tuple{String,String}[]
    default_outcome = nothing
    
    for line in rules
        if occursin("cov", line)
            continue
        end
        
        parts = split(line, ":")
        if length(parts) < 2
            continue
        end
        
        rule_part = strip(parts[2])
        parts2 = split(rule_part, "→")
        if length(parts2) < 2
            continue
        end
        
        cond = strip(parts2[1], ['(',')',' '])
        # Fix parentheses and operators
        cond = replace(cond, "˄" => " ∧ ")
        # Ensure balanced parentheses
        if !startswith(cond, "(") && occursin("∧", cond)
            cond = "(" * cond * ")"
        end
        
        out = strip(parts2[2])
        out = replace(out, "["=>"")
        out = replace(out, "]"=>"")
        out = replace(out, "'"=>"")
        
        if isempty(cond) || cond == "( )"
            default_outcome = out
            continue
        end
        
        push!(structured, (cond, out))
    end
    
    return structured, default_outcome
end

function build_rule_list(decision_list::Vector{Tuple{String,String}}, default_outcome::String)
    rules = Rule[]
    
    for (cond_str, out) in decision_list
        try
            # Ensure proper spacing around operators
            cond_str = replace(cond_str, ">" => " > ")
            cond_str = replace(cond_str, "≤" => " ≤ ")
            cond_str = replace(cond_str, "∧" => " ∧ ")
            
            φ = SoleLogics.parseformula(
                cond_str;
                atom_parser = a -> Atom(
                    parsecondition(
                        ScalarCondition,
                        a;
                        featuretype = VariableValue,
                        featvaltype = Real
                    )
                )
            )

            push!(rules, Rule(φ, out))
        catch e
            println("Warning: Failed to parse rule: $cond_str")
            println("Error: ", e)
            continue
        end
    end
    
    return rules, default_outcome
end

function convertToDecisionList(raw_rules::Vector{String})
    # Process the raw rules and extract the default outcome
    structured_rules, default_outcome = process_rules_decision_list(raw_rules)
    
    # Build the rule list and get the default outcome
    rules, default = build_rule_list(structured_rules, default_outcome)
    #rules = dnf(rules)
    # Create the DecisionList
    return DecisionList(
        rules,
        default,
        (source = "RuleCOSI", timestamp = now())
    )
end

##############################
# 4) Build ds on dnf
##############################

struct MyRule
    formula::Formula  
    outcome::String
end

##############################
# 5) FUN __init__
##############################

function rulecosiplus(ensemble::Any, X_train::Any, y_train::Any)
    
    # Convert training data to matrix format if needed
    if !isa(X_train, Matrix)
        X_train_matrix = Matrix(X_train)
    else
        X_train_matrix = X_train
    end
    
    # Generate column names
    if isa(X_train, DataFrame)
        column_names = names(X_train)
    else
        num_features = size(X_train_matrix, 2)
        column_names = ["V$i" for i in 1:num_features]
    end
    
    # Serialize the ensemble
    classes = map(String, collect(unique(y_train)))
    dict_model = serialize_julia_ensemble(ensemble, classes)
    
    # Build sklearn model
    builder = py"build_sklearn_model_from_julia"
    base_ensemble = builder(dict_model)
    
    println("======================")
    println("Ensemble serialize:", dict_model)
    println("======================")
    
    num_estimators = pycall(pybuiltin("len"), Int, base_ensemble["estimators_"])
    println("number of trees in base_ensemble:", num_estimators)
    
    n_samples = size(X_train_matrix, 1)
    n_classes = length(unique(y_train))
    
    println("Dataset: $n_samples campioni, $n_classes classi")
    
    rc = rulecosi.RuleCOSIClassifier(
        base_ensemble=base_ensemble,
        metric="roc_auc",         
        n_estimators=50,          
        tree_max_depth=15,        
        conf_threshold=0.0,       
        cov_threshold=0.0,        
        rule_order="conf",        
        early_stop=0,             
        random_state=3,
        column_names=column_names,
        verbose=2
    )
    
    
    @time rc.fit(X_train_matrix, Vector(y_train))
    
    
    max_rules = max(20, n_classes * 5)  
    
    raw_rules = py"get_simplified_rules"(rc, max_rules, 1)
    
    println("Regole grezze:")
    for r in raw_rules
        println(r)
    end
       
    dl = convertToDecisionList(raw_rules)

    return dl
end


# Required modules (adjust paths as needed)
include("utilsForTest.jl")  # The file with your 'learn_and_convert', etc.
include("apiIntrees.jl")    # The file with the 'convert_classification_rules' function
include("apiRuleCosi.jl")
