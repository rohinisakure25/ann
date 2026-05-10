import numpy as np
X = np.array([48,49,50,51,52,53,54,55,56,57])
X_bit = X % 2         
Y = np.array([0,1,0,1,0,1,0,1,0,1])   
w = 0.0
b = 0.0
eta = 0.1
for epoch in range(100):
    for i in range(len(X_bit)):
        yin = w * X_bit[i] + b
        y_hat = 1 if yin >= 0 else 0
        error = Y[i] - y_hat
        w = w + eta * error * X_bit[i]
        b = b + eta * error

num = int(input("Enter number (0-9): "))
ascii_val = ord(str(num))
last_bit = ascii_val % 2        
yin = w * last_bit + b
if yin >= 0:
    print(num, "is ODD")
else:
    print(num, "is EVEN")