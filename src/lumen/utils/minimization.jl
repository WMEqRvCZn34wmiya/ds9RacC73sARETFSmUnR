function minimizza_dnf(_minimization_scheme::Val, formula::TwoLevelDNFFormula, kwargs...)
    error("Unknown minimization scheme: $(_minimization_scheme)!")
end

# using Infiltrator
function minimizza_dnf(
    ::Val{:mitespresso},
    formula::TwoLevelDNFFormula;
    silent = true,
    mitespresso_kwargs...,
)
    formula = convert(SoleLogics.DNF, formula)
    silent || (println(); @show formula)
    formula = SoleData.espresso_minimize(formula, silent; mitespresso_kwargs...)
    silent || (println(); @show formula)
    # @infiltrate
    # @show syntaxstring(formula)
    #formula = convert(TwoLevelDNFFormula, formula) # TODO FOR NOW WE USE BYPASS... FIX THIS WHEN WE KNOW HOW TO CONVERT 
    silent || (println(); @show formula)
    return formula
end


"""
	minimizza_dnf(::Val{:espresso}, formula::TwoLevelDNFFormula, horizontal = 1.0)

Simplifies a custom OR formula using the Quine algorithm.

This function takes a `TwoLevelDNFFormula` object and applies the Quine algorithm to minimize the number of combinations in the formula. It returns a new `TwoLevelDNFFormula` object with the simplified combinations.

"""
function minimizza_dnf(::Val{:espresso}, formula::TwoLevelDNFFormula; minimization_method_kwargs...)
    # Convert TritVectors to the internal representation used by Espresso
    # We'll map: 1 -> 1, 0 -> 0, -1 -> 0 (since we only care about positive terms)
    terms = Vector{Vector{Int}}()
    for tritvec in eachcombination(formula)
        term = Vector{Int}()
        for i in 1:length(tritvec)
            val = tritvec[i]
            push!(term, val == 1 ? 1 : 0)
        end
        push!(terms, term)
    end

    function copre(cube1, cube2)
        for (b1, b2) in zip(cube1, cube2)
            if b1 != -1 && b2 != -1 && b1 != b2
                return false
            end
        end
        return true
    end

    function possono_combinarsi(cube1, cube2)
        diff_count = 0
        diff_pos = -1

        for (i, (b1, b2)) in enumerate(zip(cube1, cube2))
            if b1 != b2
                diff_count += 1
                diff_pos = i
                if diff_count > 1
                    return false, -1
                end
            end
        end

        return diff_count == 1, diff_pos
    end

    function combina_cubi(cube1, cube2, pos)
        result = copy(cube1)
        result[pos] = -1
        return result
    end

    function trova_combinazioni(cubes)
        if isempty(cubes)
            return Vector{Vector{Int}}()
        end

        result = Set{Vector{Int}}()
        combined = Set{Vector{Int}}()
        current_cubes = copy(cubes)

        while true
            found_new = false
            for i = 1:length(current_cubes)
                for j = (i+1):length(current_cubes)
                    can_combine, pos = possono_combinarsi(current_cubes[i], current_cubes[j])
                    if can_combine
                        nuovo_cubo = combina_cubi(current_cubes[i], current_cubes[j], pos)
                        if nuovo_cubo ∉ result
                            push!(result, nuovo_cubo)
                            push!(combined, current_cubes[i])
                            push!(combined, current_cubes[j])
                            found_new = true
                        end
                    end
                end
            end

            for cube in current_cubes
                if cube ∉ combined
                    push!(result, cube)
                end
            end

            if !found_new || length(result) == 0
                break
            end

            current_cubes = collect(result)
            empty!(result)
            empty!(combined)
        end

        if isempty(result)
            return current_cubes
        end

        return collect(result)
    end

    function find_essential_cubes(cubes, terms)
        if isempty(cubes) || isempty(terms)
            return terms
        end

        essential = Vector{Vector{Int}}()
        remaining_terms = Set(terms)

        while !isempty(remaining_terms)
            best_cube = nothing
            max_coverage = 0

            for cube in cubes
                coverage = count(term -> copre(cube, term), collect(remaining_terms))
                if coverage > max_coverage
                    max_coverage = coverage
                    best_cube = cube
                end
            end

            if best_cube === nothing || max_coverage == 0
                append!(essential, collect(remaining_terms))
                break
            end

            push!(essential, best_cube)
            filter!(term -> !copre(best_cube, term), remaining_terms)
        end

        return essential
    end

    function espresso_minimize(terms; minimization_method_kwargs...)
        if isempty(terms)
            return Vector{Vector{Int}}()
        end

        combined_terms = trova_combinazioni(terms)
        if isempty(combined_terms)
            return terms
        end

        essential = find_essential_cubes(combined_terms, terms)
        if isempty(essential)
            return terms
        end

        return essential
    end

    minimized_terms = espresso_minimize(terms; minimization_method_kwargs...)

    nuove_combinazioni = TritVector[]
    seen = Set{TritVector}()  

    for term in minimized_terms
        trit_combo = TritVector(nuberofatoms(formula))
        
        for (i, val) in enumerate(term)
            if val == 1
                trit_combo[i] = 1
            elseif val == -1
                trit_combo[i] = -1
            else
                trit_combo[i] = 0
            end
        end

        if trit_combo ∉ seen
            push!(seen, trit_combo)
            push!(nuove_combinazioni, trit_combo)
        end
    end

    sort!(nuove_combinazioni)  # Assuming you have defined sorting for TritVector
    return TwoLevelDNFFormula(
        nuove_combinazioni,
        nuberofatoms(formula),
        eachthresholdsbyfeature(formula),
        eachatomsbyfeature(formula),
    )
end
