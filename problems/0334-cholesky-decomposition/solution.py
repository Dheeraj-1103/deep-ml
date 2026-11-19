import numpy as np

def cholesky_decomposition(A):
    try:
        A = np.array(A, dtype=float)

        # Check if matrix is valid
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return -1

        n = A.shape[0]

        if n == 0 or not np.all(np.isfinite(A)):
            return -1

        # Check symmetry
        if not np.allclose(A, A.T):
            return -1

        L = np.zeros((n, n))

        for i in range(n):

            # Calculate diagonal element
            val = A[i, i] - sum(L[i, k]**2 for k in range(i))

            if val <= 0:
                return -1

            L[i, i] = np.sqrt(val)

            # Calculate elements below diagonal
            for j in range(i + 1, n):

                val = A[j, i] - sum(
                    L[j, k] * L[i, k] for k in range(i)
                )

                L[j, i] = val / L[i, i]

        return L.tolist()

    except (ValueError, TypeError, OverflowError):
        return -1