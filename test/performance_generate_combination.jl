using DecisionTree: load_data
using DataFrames
using Test
using TestSole
using SoleData

X, y = load_data("iris")
X = Float64.(X)
X_df = DataFrame(X, :auto)

myalphabet = alphabet(
    scalarlogiset(X_df; allow_propositional = true),
    test_operators = [<],
    force_i_variables = true,
)

myatoms = collect(atoms(myalphabet))
atoms_by_feature = @test_nowarn TestSole.group_atoms_by_feature(myatoms)

thresholds_by_feature = Dict(
    subalpha.featcondition[1].feature.i_variable => sort(subalpha.featcondition[2]) for
    subalpha in myalphabet.subalphabets
)

num_atoms = length(myatoms)
# vertical = 0.001
# num_combinations = BigInt(round(BigInt(2)^num_atoms * vertical))
num_combinations = BigInt(1000)

println(num_combinations)
@benchmark for i = 0:(num_combinations-1)
    (i % 1000) == 0 && print("-")
    combination, has_contradiction = TestSole.generate_combination_ott(
        BigInt(i),
        num_atoms,
        thresholds_by_feature,
        atoms_by_feature,
    )
    # combination, has_contradiction = TestSole.generate_combination(BigInt(i), num_atoms, thresholds_by_feature, atoms_by_feature)
end


myalphabet = alphabet(scalarlogiset(X_df; allow_propositional = true))
myatoms = atoms(myalphabet)
atoms_by_feature = @test_nowarn TestSole.group_atoms_by_feature(myatoms)
