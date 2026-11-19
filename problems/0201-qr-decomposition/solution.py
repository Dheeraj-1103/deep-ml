import numpy as np

def qr_decomposition(A):
    A = np.array(A, dtype=float)

    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):

        # Take current column
        v = A[:, j].copy()

        # Remove projections onto previous Q columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        # Calculate magnitude
        R[j, j] = np.linalg.norm(v)

        # Normalize
        if R[j, j] < 1e-10:
            raise ValueError("Columns are linearly dependent")

        Q[:, j] = v / R[j, j]

    return Q, R