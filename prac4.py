import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0,1],[0,0],[1,0],[1,1]])
Y=np.array([1,0,1,1])
w=np.zeros(2)
b=0
eta=0.1
for epoch in range(10):
    for i in range(len(X)):
        yin = np.dot(w,X[i])+b
        if yin>=0:
            y_hat=1
        else:
            y_hat=0
        w=w+eta*(Y[i]-y_hat)*X[i]
        b=b+eta*(Y[i]-y_hat)
print("Final weights:", w, "Bias:", b)
for i in range(len(X)):
    if(Y[i]==1):
        plt.scatter(X[i][0],X[i][1],color='green',marker='o',s=100,label='Class 1' if i==3 else " ")
    else:
        plt.scatter(X[i][0],X[i][1],color='red',marker='x',s=100,label='Class 0' if i==1 else " ")
x1 = np.linspace(-0.5, 1.5, 100)
x2 = -(w[0] * x1 + b) / w[1]
plt.plot(x1, x2, color='blue', label='Decision Boundary')
plt.xlim(-0.5, 1.5)   # zoom to where points are
plt.ylim(-0.5, 1.5)   # zoom to where points are
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Learning - Decision Region")
plt.legend()
plt.grid(True)
plt.show()
