import numpy as np


class Embedding():
    """
    Computes the Embedding defined in [1]

    [1]: Streaming algorithms for embedding and computing edit distance in the low distance regime
    D. Chakraborty, E. Goldenberg, M. Koucky
    STOC 16
    """
    def __init__(self, x, random_bits):
        """
        """
        # If the input is already given in binary, then we keep it as is
        if set(x) <= {'0', '1'}:
            self.s = x
        else:
            self.s = ''.join(format(ord(u), 'b') for u in x)
        self.loutput = 3 * len(self.s)
        self.random_bits = random_bits.astype(int)
        assert random_bits.shape[0] > self.loutput

    def compute(self):
        i = 0
        res = np.zeros(self.loutput, dtype=int)
        for t in range(self.loutput):
            if i < len(self.s):
                res[t] = self.s[i]
                a, b = self.random_bits[t], int(self.s[i])
                i += a * b + (1 - a) * (1 - b)
            else:
                res[t] = 0
        self.res = res
        return res


def pair_embed(x, y, random_bits):
    """
    >>> np.random.seed(42)
    >>> random_bits = np.random.randint(2, size=100)
    >>> s1 = "1" * 10 + "0" * 5
    >>> s2 = "1" * 11 + "0" * 4
    >>> pair_embed(s1, s2, random_bits)
    1
    >>> pair_embed("bad", "boy", random_bits)
    16
    """
    e1 = Embedding(x, random_bits)
    e2 = Embedding(y, random_bits)
    return sum(np.abs(e1.compute() - e2.compute()))
