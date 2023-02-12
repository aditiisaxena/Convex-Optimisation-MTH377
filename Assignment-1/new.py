import numpy as np
import numdifftools as nd

def f(x):
    return x[0] + np.log(x[1]**9)

def line_search_subroutine(f, x, d, beta=0.5, alpha=0.1):
    gradient = nd.Gradient(f)
    t = 1
    while f(x) - f(x + t * d) < alpha * t * gradient(x).T @ d:
        t = beta * t
    return t

x = np.array([2, 3])
d = np.array([-1, -1])
stepsize = line_search_subroutine(f, x, d)
print(f"The stepsize is: {stepsize}")