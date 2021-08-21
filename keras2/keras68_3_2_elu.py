import numpy as np
import matplotlib.pyplot as plt

def elu(x, a):
    return (x>0)*x + (x<=0)*(a*(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)
y = elu(x, 1)

plt.plot(x, y)
plt.grid()
plt.show()


'''
elu
: 

'''

