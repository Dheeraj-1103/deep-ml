import numpy as np

def min_norm_solution(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    M = A @ A.T

    y = np.linalg.solve(M, b)

    return A.T @ y


def min_norm_via_kkt(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    m, n = A.shape

    I = np.eye(n)
    Z = np.zeros((m, m))

    K = np.block([
        [2 * I, A.T],
        [A, Z]
    ])

    rhs = np.concatenate([
        np.zeros(n),
        b
    ])

    result = np.linalg.solve(K, rhs)

    x = result[:n]
    lam = result[n:]

    return x, lam


def min_norm_from_particular(A, x_p):
    A = np.asarray(A, dtype=float)
    x_p = np.asarray(x_p, dtype=float)

    M = A @ A.T

    y = np.linalg.solve(M, A @ x_p)

    return A.T @ y