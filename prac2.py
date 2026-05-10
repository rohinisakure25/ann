# McCulloch-Pitts Neural Network for ANDNOT Function
inputs = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]
# Weights
w1 = 1
w2 = -1

# Threshold
theta = 1

print("X1  X2  Yin  Output")

# Processing inputs
for x1, x2 in inputs:

    # Calculate activation value
    yin = (x1 * w1) + (x2 * w2)

    # Apply threshold
    if yin >= theta:
        y = 1
    else:
        y = 0

    # Display result
    print(x1, " ", x2, " ", yin, "   ", y)