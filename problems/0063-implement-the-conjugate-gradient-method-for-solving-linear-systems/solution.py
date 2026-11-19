import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Initialize solution
    if x0 is None:
        x = np.zeros(len(b))
    else:
        x = np.array(x0, dtype=float)

    # Initial residual
    r = b - A @ x

    # Initial search direction
    p = r.copy()

    for i in range(n):

        # Calculate alpha
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)

        # Update solution
        x = x + alpha * p

        # Update residual
        r_new = r - alpha * Ap

        # Check convergence
        if np.linalg.norm(r_new) < tol:
            break

        # Calculate beta
        beta = np.dot(r_new, r_new) / np.dot(r, r)

        # Update search direction
        p = r_new + beta * p

        # Update residual
        r = r_new

    return x.tolist()