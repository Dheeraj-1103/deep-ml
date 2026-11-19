import numpy as np

def woodbury_solve(sigma_diag, X, B):

    # Inverse of diagonal Sigma
    sigma_inv = 1.0 / sigma_diag

    # Calculate Sigma^-1 B
    sigma_inv_B = sigma_inv[:, None] * B if B.ndim == 2 else sigma_inv * B

    # Calculate Sigma^-1 X
    sigma_inv_X = sigma_inv[:, None] * X

    # Construct the smaller D x D matrix
    D = X.shape[1]

    middle = np.eye(D) + X.T @ sigma_inv_X

    # Calculate X.T @ Sigma^-1 @ B
    rhs = X.T @ sigma_inv_B

    # Solve the smaller linear system
    correction = np.linalg.solve(middle, rhs)

    # Apply Woodbury formula
    result = sigma_inv_B - sigma_inv_X @ correction

    return result


def sherman_morrison(A_inv, u, v):

    # Calculate A^-1 u
    y = A_inv @ u

    # Calculate v.T A^-1
    z = v @ A_inv

    # Calculate denominator
    denominator = 1.0 + v @ y

    # Check for singular update
    if denominator == 0:
        raise ValueError("Updated matrix is singular")

    # Sherman-Morrison formula
    result = A_inv - np.outer(y, z) / denominator

    return result


def logdet_rank_one_update(logdet_A, A_inv, u, v):

    # Calculate A^-1 u
    y = A_inv @ u

    # Calculate determinant update factor
    factor = 1.0 + v @ y

    # Updated determinant must be positive
    if factor <= 0:
        raise ValueError("Updated determinant is not positive")

    # Matrix determinant lemma in log space
    result = logdet_A + np.log(factor)

    return float(result)