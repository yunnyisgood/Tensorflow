import numpy as np
import matplotlib.pyplot as plt

def reaky_relu(x):
    return np.maximum(0.01*x, x) # 0보다 작은 경우, 0에 근접하는 매우 작은 값으로 변환되도록 한다 
x = np.arange(-5, 5, 0.1)
y = reaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()


'''
과제 

elu, selu, reaky relu ....
68_3_2, 3, 4로 만들 것

'''

