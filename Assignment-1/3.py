import numpy as np

def g1(x1, x2):
    return 2*x2 - x1

def g2(x1, x2):
    return 2*x1 - x2

def g3(x1, x2):
    return 1 - x1 - x2

def grad_Psi(x1, x2):
    grad_x1 = -1 / g1(x1, x2) - 1 / g2(x1, x2) + 1 / g3(x1, x2)
    grad_x2 = -2 / g1(x1, x2) + 2 / g2(x1, x2) + 1 / g3(x1, x2)
    return grad_x1, grad_x2

def gradient_descent(x1, x2, learning_rate=0.1, num_iterations=100):
    for i in range(num_iterations):
        grad_x1, grad_x2 = grad_Psi(x1, x2)
        x1 = x1 - learning_rate * grad_x1
        x2 = x2 - learning_rate * grad_x2
    return x1, x2

x1, x2 = 0.25, 0.25
x1_center, x2_center = gradient_descent(x1, x2)
print(f"Analytic center: x1={x1_center:.3f}, x2={x2_center:.3f}")
