import itertools as it
import operator as op
from collections import defaultdict
import numpy as np


# Here we encode tensors over F_2 using the bits in an int64 - this is a tensint
# NOTE: to add tensints, use ^ (XOR)!!


def tensint_get(t, i):
    """Return true if ith coordinate of tensor in int64 form is 1."""
    return t & (1 << i) > 0


def tensint_get3(t, N, i, j, k):
    """Return true if (i, j, k)th coordinate of tensor in int64 form is 1."""
    l = i * N * N + j * N + k
    return t & (1 << l) > 0


def str_of_tensint3(t, N):
    def get_elem(i, j, k):
        return '1' if tensint_get3(t, N, i, j, k) else '0'
    return '\n\n'.join([
        '\n'.join(
            [' '.join([get_elem(i, j, k) for k in xrange(N)])
                for j in xrange(N)]) for i in xrange(N)])


def mk_e(i, N):
    """Return e_i in F_2^N."""
    assert 0 <= i and i < N
    return np.int64(0 + (1 << i))


def mk_e3(i, j, k, N):
    """Return e_i \otimes e_j \otimes e_k in (F_2^N)^{\otimes 3}."""
    return mk_e(i * N * N + j * N + k, N ** 3)


def mk_per3():
    """Returns per_3 from (F_2^3)^{\otimes 3}."""
    return reduce(op.xor,
                  (mk_e3(i, j, k, 3)
                   for (i, j, k) in it.permutations([0, 1, 2])),
                  np.int64(0))


def tensint_of_vector(v):
    """Converts a vector in F_2^N to a tensint in F_2^N."""
    return reduce(op.xor, (mk_e(i, len(v)) for i in xrange(len(v)) if v[i] != 0),
                  np.int64(0))


def tensor_product(t1, t2, t3):
    """Given 3 vectors in F_2^N, returns t1 \otimes t2 \otimes t3."""
    N = len(t1)
    res = np.int64(0)
    for i, j, k in it.product(xrange(N), repeat=3):
        if t1[i] and t2[j] and t3[k]:
            res ^= mk_e3(i, j, k, N)
    return res


def simple_tensors(N):
    """Returns the non-zero tensors in (F_2^N)^{\otimes 3}."""
    vectors = [np.array(x) for x in it.product([True, False], repeat=N)][:-1]

    return [tensor_product(vectors[i], vectors[j], vectors[k])
            for i, j, k in it.product(xrange(len(vectors)), repeat=3)
            ]
