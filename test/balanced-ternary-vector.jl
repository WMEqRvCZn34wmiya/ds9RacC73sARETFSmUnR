using TestSole: @bt_str

a = bt"+-0++0+"
println("a: $(Int(a)), $a")
b = BalancedTernaryVector(-436)
println("b: $(Int(b)), $b")
c = BalancedTernaryVector("+-++-")
println("c: $(Int(c)), $c")
r = a * (b - c)
println("a * (b - c): $(Int(r)), $r")

@assert Int(r) == Int(a) * (Int(b) - Int(c))
