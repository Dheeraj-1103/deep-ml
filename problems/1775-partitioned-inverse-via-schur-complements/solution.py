import numpy as np

def partitioned_inverse(M, k):
    n = M.shape[0]

    # Split matrix into four blocks
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    # Calculate H inverse
    H_inv = np.linalg.inv(H)

    # Calculate Schur complement: M/H = E - F H^-1 G
    S = E - F @ H_inv @ G

    # Inverse of Schur complement
    S_inv = np.linalg.inv(S)

    # Calculate four blocks of M inverse
    top_left = S_inv

    top_right = -S_inv @ F @ H_inv

    bottom_left = -H_inv @ G @ S_inv

    bottom_right = H_inv + H_inv @ G @ S_inv @ F @ H_inv

    # Combine four blocks
    result = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    return result


def block_slogdet(M, k):
    # Split into blocks
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    # Calculate Schur complement
    H_inv = np.linalg.inv(H)
    S = E - F @ H_inv @ G

    # Determinants of both blocks
    sign1, logdet1 = np.linalg.slogdet(S)
    sign2, logdet2 = np.linalg.slogdet(H)

    # Combine signs and log determinants
    sign = sign1 * sign2
    logabsdet = logdet1 + logdet2

    return sign, logabsdet


def bordered_inverse(A_inv, b, c):
    # Convert inputs to numpy arrays
    A_inv = np.asarray(A_inv, dtype=float)
    b = np.asarray(b, dtype=float)

    # Calculate A^-1 b
    y = A_inv @ b

    # Schur complement of A
    s = c - b.T @ y

    # Check whether the matrix is singular
    if abs(s) < 1e-12:
        raise ValueError("Schur complement is too close to zero")

    # Calculate four blocks
    top_left = A_inv + np.outer(y, y) / s

    top_right = -y / s

    bottom_left = -y.T / s

    bottom_right = np.array([[1.0 / s]])

    # Combine blocks
    result = np.block([
        [top_left, top_right.reshape(-1, 1)],
        [bottom_left.reshape(1, -1), bottom_right]
    ])

    return result
    