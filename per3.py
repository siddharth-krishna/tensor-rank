import itertools as it
from collections import defaultdict
import tensint as ti


def one_indices_of_tensint(t, N):
    return [(1 << i) for i in xrange(N**3) if ti.tensint_get(t, i)]


def sum_contains(b1, b2, target):
    # return any(((t1 ^ t2) == target for t1, t2 in it.product(b1, b2)))
    for t1, t2 in it.product(b1, b2):
        if (t1 ^ t2) == target:
            return (t1, t2)
    return None


def find_with_buckets(tensors1, tensors2, target, N):
    one_indices = one_indices_of_tensint(target, N)
    buckets1 = defaultdict(list)
    buckets2 = defaultdict(list)
    # Sort tensors into buckets

    def get_key_of_t(t):
        key = 0
        for i in one_indices:
            key <<= 1
            key += 1 if t & i > 0 else 0
        return key
    for t in tensors1:
        buckets1[get_key_of_t(t)].append(t)
    for t in tensors2:
        buckets2[get_key_of_t(t)].append(t)

    # Construct all keys
    all_ones = (1 << len(one_indices)) - 1

    def rev_key(key):
        return all_ones ^ key
    keys = range(all_ones + 1)
    bucket_pairs = [(buckets1[key], buckets2[rev_key(key)]) for key in keys]
    # return bucket_pairs

    for (b1, b2) in bucket_pairs:
        res = sum_contains(b1, b2, target)
        if res:
            return res
    return None


def find_per3_expr():
    """Tries to find an expression for per_3 as a sum of simple tensors."""
    simpls = ti.simple_tensors(3)
    print "Created %d simple tensors." % (len(simpls,))
    sums = set(t1 ^ t2 for (t1, t2) in it.combinations(simpls, 2))
    print "Created %d tensors of rank <= 2." % (len(sums,))
    trips = set(t1 ^ t2 for (t1, t2) in it.product(simpls, sums))
    print "Created %d tensors of rank <= 2." % (len(trips,))
    per3 = ti.mk_per3()

    print "Trying to find a rank <= 4 expression for per_3:"
    res = find_with_buckets(sums, sums, per3, 3)
    if not res:
        print "Nope."
    else:
        assert False

    print "Trying to find a rank <= 5 expression for per_3:"
    res = find_with_buckets(sums, trips, per3, 3)
    if res:
        t1, t2 = res
        t3, t4 = sum_contains(simpls, simpls, t1)
        t5, t6 = sum_contains(simpls, sums, t2)
        t7, t8 = sum_contains(simpls, simpls, t6)
        print "Found it. per_3 is the sum of the following simple tensors:"
        print "\n\n+\n\n".join([ti.str_of_tensint3(t, 3) for t in [t3, t4, t5, t7, t8]])
