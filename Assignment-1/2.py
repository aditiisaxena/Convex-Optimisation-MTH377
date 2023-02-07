import numpy as np

def f(x, y):
    return (x**2 - 3*y**2)**2 + np.sin(x**2 + y**2)**2

def grad_f(x, y):
    grad_x = 4*x*(x**2 - 3*y**2) + 2*np.sin(x**2 + y**2)*np.cos(x**2 + y**2)*2*x
    grad_y = -6*y*(x**2 - 3*y**2) + 2*np.sin(x**2 + y**2)*np.cos(x**2 + y**2)*2*y
    return grad_x, grad_y

def gradient_descent(x0, y0, learning_rate=0.1, num_iterations=100):
    x = x0
    y = y0
    for i in range(num_iterations):
        grad_x, grad_y = grad_f(x, y)
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
    return x, y

x0, y0 = 1, 1
x_min, y_min = gradient_descent(x0, y0)
print(f"Minimum at: x={x_min:.3f}, y={y_min:.3f}")
print(f"Minimum value: f(x, y)={f(x_min, y_min):.3f}")
