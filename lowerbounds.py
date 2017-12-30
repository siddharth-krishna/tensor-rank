"""
This file contains the calculations used in Sections 4 & 5.

It computes the lower bounds given by Proposition 2.6 for the
rank of the 5x5 and 7x7 permanent and determinant tensors,
and the symmetric 3x3 permanent and determinant tensors.
"""

import itertools as it
import numpy as np


def sign(xs):
    res = 1
    for i in xrange(len(xs)):
        for j in xrange(i):
            if xs[i] < xs[j]:
                res *= -1
    return res


def mk_det_5_layer(missing):
    """Return the [missing]-th layer of det_5."""
    res = np.zeros((25, 25), dtype=int)
    indices = [x for x in xrange(5) if x != missing]
    for (i1, j1, i2, j2) in it.permutations(indices):
        res[5 * i1 + i2, 5 * j1 + j2] = sign([missing, i1, j1, i2, j2])
    return res


def mk_per_5_layer(missing):
    """Return the [missing]-th layer of per_5."""
    res = np.zeros((25, 25), dtype=int)
    indices = [x for x in xrange(5) if x != missing]
    for (i1, j1, i2, j2) in it.permutations(indices):
        res[5 * i1 + i2, 5 * j1 + j2] = 1
    return res


def mk_det_7_layer(missing):
    """Return the [missing]-th layer of det_7."""
    res = np.zeros((343, 343), dtype=int)
    indices = [x for x in xrange(7) if x != missing]
    for (i1, j1, k1, i2, j2, k2) in it.permutations(indices):
        res[49 * i1 + 7 * j1 + k1, 49 * i2 + 7 * j2 +
            k2] = sign([missing, i1, j1, k1, i2, j2, k2])
    return res


def mk_per_7_layer(missing):
    """Return the [missing]-th layer of per_7."""
    res = np.zeros((343, 343), dtype=int)
    indices = [x for x in xrange(7) if x != missing]
    for (i1, j1, k1, i2, j2, k2) in it.permutations(indices):
        res[49 * i1 + 7 * j1 + k1, 49 * i2 + 7 * j2 + k2] = 1
    return res


def mk_sdet_3_layers():
    d = np.zeros((9, 9, 9))
    for (i, j, k) in it.permutations([0, 1, 2]):
        for (l, m, n) in it.permutations([i, 3 + j, 6 + k]):
            d[l][m][n] = sign([i, j, k])

    return d


def mk_sper_3_layers():
    d = np.zeros((9, 9, 9))
    for (i, j, k) in it.permutations([0, 1, 2]):
        for (l, m, n) in it.permutations([i, 3 + j, 6 + k]):
            d[l][m][n] = 1
    return d


def mk_Ls(p):
    """Returning [L(e_i) for 1 <= i <= 2 * p + 1]."""
    y = list(it.combinations(range(2 * p + 1), p))
    z = list(it.combinations(range(2 * p + 1), p + 1))
    q = len(y)
    b = [np.zeros((q, q)) for k in range(2 * p + 1)]

    for h in range(2 * p + 1):
        for i in range(q):
            t = sorted(y[i] + (h,))
            k = 1
            for j in range(p):
                if h > y[i][j]:
                    k = k * (-1)
            for m in range(q):
                if t == list(z[m]):
                    b[h][i][m] = k
    return b

if __name__ == "__main__":
    Ls_5 = mk_Ls(2)
    Ls_7 = mk_Ls(3)
    Ls_9 = mk_Ls(4)

    m = np.sum(np.kron(Ls_5[i], mk_per_5_layer(i)) for i in xrange(5))
    print "Lower bound for per_5: %f" % (np.linalg.matrix_rank(m) / 6.0,)

    m = np.sum(np.kron(Ls_5[i], mk_det_5_layer(i)) for i in xrange(5))
    print "Lower bound for det_5: %f" % (np.linalg.matrix_rank(m) / 6.0,)

    m = np.sum(np.kron(Ls_7[i], mk_per_7_layer(i)) for i in xrange(7))
    print "Lower bound for per_7: %f" % (np.linalg.matrix_rank(m) / 20.0,)

    m = np.sum(np.kron(Ls_7[i], mk_det_7_layer(i)) for i in xrange(7))
    print "Lower bound for det_7: %f" % (np.linalg.matrix_rank(m) / 20.0)

    sdet = mk_sdet_3_layers()
    m = np.sum(np.kron(Ls_9[i], sdet[i]) for i in xrange(9))
    print "Lower bound for sdet_3: %f" % (np.linalg.matrix_rank(m) / 70.0, )

    sper = mk_sper_3_layers()
    m = np.sum(np.kron(Ls_9[i], sper[i]) for i in xrange(9))
    print "Lower bound for sdet_3: %f" % (np.linalg.matrix_rank(m) / 70.0, )
