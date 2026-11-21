import numpy as np
def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    L=np.array(L)
    V=np.dot(v,L)
    u=np.dot(L,L)
    pro=float(V/u)
    res=pro*L
    return np.round(res,3).tolist()

