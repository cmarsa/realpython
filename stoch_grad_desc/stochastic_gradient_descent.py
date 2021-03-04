# -*- coding: utf-8 -*-
# stoch_grad_desc/stochastic_gradient_descent.py
"""
Stochastic Gradient Descent
---------------------------
Stochastic gradient descent algorithms are a modification of gradient descent.
In stochastic gradient descent, you calculate the gradient using just a
random small part of the observations instead of all of them.
In some cases, this approach can reduce computation time.

Online stochastic gradient descent is a variant of stochastic gradient
descent in which you estimate the gradient of the cost function for
each observation and update the decision variables accordingly.
This can help you find the global minimum, especially if the
objective function is convex.

Batch stochastic gradient descent is somewhere between ordinary
gradient descent and the online method. The gradients are
calculated and the decision variables are updated iteratively
with subsets of all observations, called minibatches. This variant
is very popular for training neural networks.

You can imagine the online algorithm as a special kind of batch
algorithm in which each minibatch has only one observation.
Classical gradient descent is another special case in which
there’s only one batch containing all observations.


Minibatches in Stochastic Gradient Descent
------------------------------------------
As in the case of the ordinary gradient descent, stochastic gradient descent starts with an initial vector of decision variables and updates it through several iterations. The difference between the two is in what happens inside the iterations:

    Stochastic gradient descent randomly divides the set of observations into minibatches.
    For each minibatch, the gradient is computed and the vector is moved.
    Once all minibatches are used, you say that the iteration, or epoch, is finished and start the next one.


Momentum in Stochastic Gradient Descent
---------------------------------------
The learning rate can have a significant impact on the result of gradient descent.
You can use several different strategies for adapting the learning rate during
the algorithm execution. You can also apply momentum to your algorithm.

You can use momentum to correct the effect of the learning rate.
The idea is to remember the previous update of the vector and apply
it when calculating the next one. You don’t move the vector exactly
in the direction of the negative gradient, but you also tend to keep
the direction and magnitude from the previous move.

The parameter called the decay rate or decay factor defines
how strong the contribution of the previous update is. 
"""
import numpy as np
from gradient_descent import ssr_gradient

def stochastic_gradient_descent(gradient, x, y, start, learn_rate=0.1, batch_size=1, n_iter=50,
                                tolerance=1e-06, dtype='float64', random_state=None):
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
        batch_size {int}: number of observations in each minibatch
        tolerance {float}: minimal allowed movement in each iteration
        dtype {type}: data type for the arrays
        random_state {int}: a seed for the random number generator
    
    returns
    -------
        vector {tuple, list, array}: optimized point (convergence is reached) 
    """
    # checking if the gradient is callable
    if not callable(gradient):
        raise TypeError('`gradient` must be callable')

    # setting up the data type for numpy arrays
    dtype_ = np.dtype(dtype)

    # converting x and y to numpy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError('`x` and `y` lengths do not match')
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    # initializing the random number generator
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # initializing the values of the variables
    vector = np.array(start, dtype=dtype_)

    # setting up and checking in the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError('`learn_rate` must be greatet than zero')

    # setting up and checking the size of minibatches
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError('`batch_size` must be greater than zerp and less'
                         + 'than or equal to the number of observations')
    
    # setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError('`n_iter` must be greater than zero')

    # performing the gradient descent loop
    for __ in range(n_iter):
        # shuffle x and y
        rng.shuffle(xy)

        # minimatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
            diff = -learn_rate * grad

            # checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # update the values of hte variables
            vector += diff
    
    return vector if vector.shape else vector.item()


def stochastic_gradient_descent_momentum(gradient, x, y, start, learn_rate=0.1, decay_rate=0.0,
                                         batch_size=1, n_iter=50, tolerance=1e-06, dtype='float64',
                                         random_state=None):
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
        decay_rate {float}: between 0 and 1, strength of the contribution of the previous update
            during the iteration/epoch.
        n_iter {int}: number of iterations
        batch_size {int}: number of observations in each minibatch
        tolerance {float}: minimal allowed movement in each iteration
        dtype {type}: data type for the arrays
        random_state {int}: a seed for the random number generator
    
    returns
    -------
        vector {tuple, list, array}: optimized point (convergence is reached) 
    """
    # checking if the gradient is callable
    if not callable(gradient):
        raise TypeError('`gradient` must be callable')

    # setting up the data type for numpy arrays
    dtype_ = np.dtype(dtype)

    # converting x and y to numpy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError('`x` and `y` lengths do not match')
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    # initializing the random number generator
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # initializing the values of the variables
    vector = np.array(start, dtype=dtype_)

    # setting up and checking in the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError('`learn_rate` must be greatet than zero')

    # setting up and checking the decay rate
    decay_rate = np.array(decay_rate, dtype=dtype_)
    if np.any(decay_rate < 0) or np.any(decay_rate > 1):
        raise ValueError('`decay_rate` must be between zero and one')

    # setting up and checking the size of minibatches
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError('`batch_size` must be greater than zerp and less'
                         + 'than or equal to the number of observations')
    
    # setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError('`n_iter` must be greater than zero')

    # setting the difference to zero for the first iteration
    diff = 0
    # performing the gradient descent loop
    for __ in range(n_iter):
        # shuffle x and y
        rng.shuffle(xy)

        # minimatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
            diff = decay_rate * diff -learn_rate * grad
            """
            decay_rate * diff is the momentum, or impact of the previous move.
            -learn_rate * grad is the impact of the current gradient.
            """

            # checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # update the values of hte variables
            vector += diff
    
    return vector if vector.shape else vector.item()

if __name__ == '__main__':
    # SGD for SSR
    x = np.array([5, 15, 25, 35, 45, 55])
    y = np.array([5, 20, 14, 32, 22, 38])
    opt = stochastic_gradient_descent(ssr_gradient, x, y, start=[0.5, 0.5],
                                      learn_rate=0.0008, batch_size=3,
                                      n_iter=100_000, random_state=0)
    print(opt)
