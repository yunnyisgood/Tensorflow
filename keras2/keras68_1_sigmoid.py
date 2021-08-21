import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # np.exp -> 지수함수로 변환 

x = np.arange(-5, 5, 0.1) 
print(x) # 100개

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()
