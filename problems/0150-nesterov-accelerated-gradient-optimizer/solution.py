import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    # Your code here
    lookahead = parameter - velocity*momentum
    gradent = grad_fn(lookahead)
    new_velocity = velocity*momentum + learning_rate*gradent
    new_parameter = parameter-new_velocity
    return np.round(new_parameter, 5), np.round(new_velocity, 5)