import numpy as np


class Edit():
    """
    Computes the edit distance between s1 and s2
    using a classical dynamic programming approach.
        >>> Edit("a", "b").distance()
        1
        >>> Edit("aa", "a").distance()
        1
        >>> Edit("abcdef", "abdef").distance()
        1
        >>> Edit("scissor", "sisters").distance()
        4
        >>> Edit("aaa", "aaabc").distance()
        2
        >>> Edit("bonjour", "au revoir").distance()
        7

        >>> len(Edit("aaa", "aaabc").compute_path())
        3
        >>> "bordeaux" in Edit("bourgogne", "bordeaux").compute_path()
        True
        >>> len(Edit("bonjour", "au revoir").compute_path())
        8
        >>> 'aaa aac abc bc' == " ".join(Edit("aaa", "bc").compute_path())
        True

    """
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.n = len(s1)
        self.m = len(s2)
        self.t = np.zeros((self.n + 1, self.m + 1), dtype=int)
        self.t[0] = range(self.m + 1)
        self.t[:, 0] = range(self.n + 1)

        # The following boolean flag is used to determine whether
        # compute_edit_array has already been called
        self.computed = False

    def compute_edit_array(self):
        """
        Computes the array of distances between
        any prefix of s1 and prefix of s2 using DP
        """
        self.computed = True
        for i in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                if self.s1[i - 1] == self.s2[j - 1]:
                    self.t[i][j] = self.t[i - 1][j - 1]
                else:
                    self.t[i][j] = 1 + min(self.t[i - 1][j], self.t[i][j - 1], self.t[i - 1][j - 1])

    def distance(self):
        """
        Returns the distance between s1 and s2
        """
        if not self.computed:
            self.compute_edit_array()
        return self.t[self.n][self.m]

    def compute_path(self):
        """
        Computes the whole path of transformations
        leading to s2 from s1
        """
        if not self.computed:
            self.compute_edit_array()
        path = [self.s1]
        i, j = self.n, self.m
        while (i > 0 and j > 0):
            a, b, c = self.t[i - 1][j], self.t[i][j - 1], self.t[i - 1][j - 1]
            if a < min(b, c):
                i = i - 1
                path += [self.s1[:i] + self.s2[j:]]
            elif b < c:
                j = j - 1
                path += [self.s1[:i] + self.s2[j:]]
            else:
                if self.s1[i - 1] != self.s2[j - 1]:
                    path += [self.s1[:i - 1] + self.s2[j - 1:]]
                i, j = i - 1, j - 1
        if j > 0:
            path += [self.s2[j - k - 1:] for k in range(j)]
        if i > 0:
            path += [self.s1[:i - 1 - k] + self.s2 for k in range(i)]
        return path
