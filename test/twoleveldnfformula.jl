using SoleLogics
using Test
using TestSole
using TestSole: TwoLevelDNFFormula

num_atoms = 5
num_conjuncts = 4
thresholds_by_feature = Dict(1 => [1, 2, 3], 2 => [10, 20])
atoms_by_feature =
    Dict(1 => [(1, true), (2, false), (3, false)], 2 => [(10, true), (20, false)])
prime_mask = [ones(Bool, size(combination)) for combination in combinations]

combinations = map(x -> BitVector(x()), fill(() -> rand(Bool, num_atoms), num_conjuncts))

f = @test_nowarn TwoLevelDNFFormula(
    combinations,
    num_atoms,
    thresholds_by_feature,
    atoms_by_feature,
    prime_mask,
)
TestSole.stampa_dnf(stdout, f)

combinations = map(x -> BitVector(x()), fill(() -> fill(false, num_atoms), num_conjuncts))

f = @test_nowarn TwoLevelDNFFormula(
    combinations,
    num_atoms,
    thresholds_by_feature,
    atoms_by_feature,
    prime_mask,
)
TestSole.stampa_dnf(stdout, f)

combinations = map(x -> BitVector(x()), fill(() -> fill(true, num_atoms), num_conjuncts))

f = @test_nowarn TwoLevelDNFFormula(
    combinations,
    num_atoms,
    thresholds_by_feature,
    atoms_by_feature,
    prime_mask,
)
TestSole.stampa_dnf(stdout, f)


solef = @test_nowarn Base.convert(SoleLogics.DNF, f)
println(solef)
f2 = @test_broken Base.convert(TwoLevelDNFFormula, solef)
@test f == f2
