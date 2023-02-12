import numpy as np
import numdifftools as nd

def f(var):
    return (var[0]**2 - 3*var[1]**2)**2 + (np.sin(var[0]**2 + var[1]**2))**2

def lineSearchSubroutine(f, x, d, beta=0.5, alpha=0.1):
    x0 = x
    t = 1
    gradient = nd.Gradient(f)
    while f(x0) - f(x0 + t * d) < -t*alpha*nd.directionaldiff(f, x0, d):
        t = beta*t
    return t

def combinationDescent(f, x, tolerance):
    gradient = nd.Gradient(f)
    x0 = x
    while np.linalg.norm(gradient(x0)) > tolerance:
        hessian = nd.Hessian(f)
        evals , evecs = np.linalg.eig(hessian(x0))
        test_evals = evals >0
        if np . all( test_evals ) == True :
            d = np.linalg.inv(hessian(x0))@(-gradient(x0))
        else :
            d = -gradient(x0)
        t = lineSearchSubroutine(f, x0, d)
        x0 = x0 + t*d
    return x0

x = np.array([1.0, 1.0])
tolerance = 1e-6
print(combinationDescent(f, x, tolerance))
