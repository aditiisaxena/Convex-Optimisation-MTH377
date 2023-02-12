import numpy as np
import matplotlib.pyplot as plt

def f(x):
    
    return x**2 + np.log(x)

def L(x):
    return 3*x - 2

def Q(x):
    return x**2/2 + 2*x - 3/2

x = np.linspace(0, 2, 100)

plt.plot(x, f(x), label='f(x)')
plt.plot(x, L(x), label='L(x)')
plt.plot(x, Q(x), label='Q(x)')
plt.legend()
plt.show()

