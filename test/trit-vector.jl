# Create two ternary arrays
a = TritVector(3)
b = TritVector(3)

# Set some values
a[1] = -1
a[2] = 0
a[3] = 1

b[1] = 1
b[2] = -1
b[3] = 0

# Arithmetic operations
c = a + b    # Addition
d = a * b    # Multiplication
e = -a       # Negation

println(a)   # TritVector([-1, 0, 1])
println(b)   # TritVector([1, -1, 0])
println(c)   # TritVector([0, -1, 1])



# Create a new array of 6 trits
arr = TritVector(6)

# Set some values
arr[1] = -1
arr[2] = 0
arr[3] = 1
arr[4] = -1
arr[5] = 0
arr[6] = 1

# Check the actual memory usage
println(sizeof(arr))

# Verify values are stored correctly
for i = 1:length(arr)
    println("arr[$i] = $(arr[i])")
end
