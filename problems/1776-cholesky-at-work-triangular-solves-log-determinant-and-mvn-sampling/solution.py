import numpy as np

def forward_sub(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        s = 0

        for j in range(i):
            s += L[i, j] * y[j]

        y[i] = (b[i] - s) / L[i, i]

    return y

def back_sub(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        s = 0

        for j in range(i+1, n):
            s += U[i, j] * x[j]

        x[i] = (y[i] - s) / U[i, i]

    return x

def chol_solve(A, b):
    L = np.linalg.cholesky(A)

    y = forward_sub(L, b)

    x = back_sub(L.T, y)

    return x

def logdet_chol(A):
    L = np.linalg.cholesky(A)

    return float(2 * np.sum(np.log(np.diag(L))))

def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)

    return eps @ L.T + mu
