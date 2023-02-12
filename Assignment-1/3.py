import numpy as np
import numdifftools as nd

def g1(x):
    return 2*x[1] - x[0]

def g2(x):
    return 2*x[0] - x[1]

def g3(x):
    return 1 - x[0] - x[1]

def f(x):
    return np.log(1/(g1(x)*g2(x)*g3(x)))

def lineSearchSubroutine(f, x, d, beta=0.5, alpha=0.5):
    t = 1
    while f(x) - f(x + t * d) < -t*alpha*np.dot(nd.Gradient(f)(x), d):
        t = beta*t
    return t

def combinationDescent(f, x, tolerance):
    gradient = nd.Gradient(f)(x)
    while np.linalg.norm(gradient) > tolerance:
        hessian = nd.Hessian(f)(x)
        evals , evecs = np.linalg.eig(hessian)
        if np.all(evals > 0):
            d = -np.linalg.solve(hessian, gradient)
        else :
            d = -gradient
        t = lineSearchSubroutine(f, x, d)
        x = x + t*d
        gradient = nd.Gradient(f)(x)
    return x

x = np.array([0.25, 0.25])
tolerance = 1e-6
y = combinationDescent(f, x, tolerance)
print(f(y))