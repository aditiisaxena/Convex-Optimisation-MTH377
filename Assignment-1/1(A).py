import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + np.log(x)

def L(x, x0=1, f0=f(1), df0=2):
    return f0 + df0*(x-x0)

def Q(x, x0=1, f0=f(1), df0=2, d2f0=2):
    return f0 + df0*(x-x0) + (1/2)*d2f0*(x-x0)**2

x = np.linspace(0, 2, 100)
y_f = f(x)
y_L = L(x)
y_Q = Q(x)

plt.plot(x, y_f, label='f(x)')
plt.plot(x, y_L, label='L(x)')
plt.plot(x, y_Q, label='Q(x)')
plt.legend()
plt.show()
