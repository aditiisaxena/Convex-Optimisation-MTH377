import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + np.log(x)

def L(x, x0=1, f0=f(1), df0=2):
    return f0 + df0*(x-x0)

def Q(x, x0=1, f0=f(1), df0=2, d2f0=2):
    return f0 + df0*(x-x0) + (1/2)*d2f0*(x-x0)**2

def eL(x):
    return f(x) - L(x)

def eQ(x):
    return f(x) - Q(x)

x = np.linspace(0, 2, 100)
y_eL = eL(x) / (x-1)
y_eQ = eQ(x) / (x-1)**2

plt.plot(x, y_eL, label='eL(x)/(x-1)')
plt.plot(x, y_eQ, label='eQ(x)/(x-1)^2')
plt.legend()
plt.show()
