# -*- coding: utf-8 -*-
# stoch_grad_desc/gradient_descent.py

import numpy as np


def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    """
    arguments
    ---------
        gradient {function or callable}: function or callable object that
            takes a vector and returns the gradient of the function that
            is minimized.
        start {tuple, list, array}: point where the algorithm starts
            its search.
        learn_rate {scalar, float, int}: learning rate that controls the
            magnitude of the vector update.
        n_iter {int}: number of iterations
        tolerance {float}: minimal allowed movement in each iteration
    
    returns
    -------
        vector {tuple, list, array}: optimized point (convergence is reached) 
    """
    vector = start
    for __ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


def gradient_descent_simple_lin_reg(gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06):
    """
    arguments
    ---------
        gradient {function or callable}: function or callable object that
            takes a vector and returns the gradient of the function that
            is minimized.
        start {tuple, list, array}: point where the algorithm starts
            its search.
        learn_rate {scalar, float, int}: learning rate that controls the
            magnitude of the vector update.
        n_iter {int}: number of iterations
        tolerance {float}: minimal allowed movement in each iteration
    
    returns
    -------
        vector {tuple, list, array}: optimized point (convergence is reached) 
    """
    vector = start
    for __ in range(n_iter):
        diff = -learn_rate * np.array(gradient(x, y, vector))
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


def ssr_gradient(x, y, b):
    """Gradient function for the function SSR (Sum of Squared Residuals)
    for the simple linear regression: f(x) = b_0 + b_1*x

    arguments
    ---------
        x {array}: observation input
        y (array): observation output
        b {array} current values of the decision variables b_0 and b_1
    returns
    -------
        the partial derivatives of SSR with respect to b_0 and b_1
    """
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()

if __name__ == '__main__':
    # Small learning rates can result in very slow convergence.
    # If the number of iterations is limited, then the algorithm
    # may return before the minimum is found. Otherwise,
    # the whole process might take an unacceptably large amount of time. 
    start = 10.0
    ethas = [0.001, 0.01, 0.2, 0.8, 1, 2, 10, 110]
    for etha in ethas:
        optimum = gradient_descent(lambda v: 2*v, 10.0, etha)
        print(f'lr: {etha}  optimum: {optimum}')

    # Nonconvex functions might have local minima or saddle
    # points where the algorithm can get trapped.
    # In such situations, your choice of learning rate or starting
    # point can make the difference between finding a local minimum 
    # and finding the global minimum.
    # Consider the function 𝑣⁴ - 5𝑣² - 3𝑣.
    # It has a global minimum in 𝑣 ≈ 1.7 and a local minimum in 𝑣 ≈ −1.42.
    # The gradient of this function is 4𝑣³ − 10𝑣 − 3.
    # Let’s see how gradient_descent() works here:
    start = 10.0
    ethas = [0.001, 0.01, 0.2]
    for etha in ethas:
        optimum = gradient_descent(lambda v: 4 * v**3 - 10 * v - 3, 0.0, etha)
        print(f'lr: {etha}  optimum: {optimum}')
    
    # Besides the learning rate, the starting point can affect
    # the solution significantly, especially with nonconvex functions.


    # Ordinary Least Squares
    # Now apply your new version of gradient_descent() to find the
    # regression line for some arbitrary values of x and y:
    x = np.array([5, 15, 25, 35, 45, 55])
    y = np.array([5, 20, 14, 32, 22, 38])
    optimum = gradient_descent_simple_lin_reg(ssr_gradient, x, y, start=[0.5, 0.5],
                                              learn_rate=0.0008, n_iter=100_000)
    print('OLS optimization: ', optimum)
    
    # The result is an array with two values that correspond to the decision
    # variables: b_0 = 5.63 and b_1 = 0.54. The best regression line
    # is f(x) = 5.63 + 0.54*sx. As in the previous examples, this result
    # heavily depends on the learning rate. You might not get such a good
    # result with too low or too high of a learning rate.

    