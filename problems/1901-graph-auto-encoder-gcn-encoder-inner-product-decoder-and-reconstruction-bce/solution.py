import numpy as np


def gae_encode(A, X, W1, W2):

    N = A.shape[0]

    A_tilde = A + np.eye(N)

    D = np.sum(A_tilde, axis=1)

    D_inv = np.diag(1 / np.sqrt(D))

    A_hat = D_inv @ A_tilde @ D_inv

    H = A_hat @ X @ W1

    H = np.maximum(0, H)

    Z = A_hat @ H @ W2

    return Z


def gae_logits(Z, pairs):

    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)

    return np.sum(Z[pairs[:, 0]] * Z[pairs[:, 1]], axis=1)


def gae_loss(A, Z, pos_weight=None):

    N = A.shape[0]

    logits = Z @ Z.T

    if pos_weight is None:

        positives = np.sum(A)

        if positives == 0:
            pos_weight = 1.0
        else:
            pos_weight = (N*N - positives) / positives

    log_pos = -np.logaddexp(0, -logits)

    log_neg = -np.logaddexp(0, logits)

    loss = -(pos_weight * A * log_pos + (1 - A) * log_neg)

    return float(np.mean(loss))


def sampled_gae_loss(Z, pos_pairs, neg_pairs):

    pos_pairs = np.asarray(pos_pairs, dtype=int).reshape(-1, 2)
    neg_pairs = np.asarray(neg_pairs, dtype=int).reshape(-1, 2)

    pos_logits = np.sum(
        Z[pos_pairs[:, 0]] * Z[pos_pairs[:, 1]], axis=1
    )

    neg_logits = np.sum(
        Z[neg_pairs[:, 0]] * Z[neg_pairs[:, 1]], axis=1
    )

    pos_loss = np.logaddexp(0, -pos_logits)

    neg_loss = np.logaddexp(0, neg_logits)

    return float(np.mean(np.concatenate([pos_loss, neg_loss])))


def vgae_sample(mu, logvar, eps):

    return mu + np.exp(0.5 * logvar) * eps


def vgae_kl(mu, logvar):

    kl = -0.5 * np.sum(
        1 + logvar - mu**2 - np.exp(logvar),
        axis=1
    )

    return float(np.mean(kl))