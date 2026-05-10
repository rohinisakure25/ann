import numpy as np

x1 = np.array([1,1,1,-1])
y1 = np.array([1,-1])

x2 = np.array([-1,-1,1,1])
y2 = np.array([-1,1])

# Weight matrix
W = np.outer(y1, x1) + np.outer(y2, x2)

# X -> Y
def bam_forward(x):
    y = np.dot(W, x)
    return np.where(y >= 0, 1, -1)

# Y -> X
def bam_backward(y):
    x = np.dot(W.T, y)
    return np.where(x >= 0, 1, -1)

# Test
x_test = np.array([1,-1,-1,-1])

y_out = bam_forward(x_test)
x_recall = bam_backward(y_out)

print("Input x:", x_test)
print("Recalled y:", y_out)
print("Recalled x:", x_recall)