from edit import Edit
from embedding import Embedding, pair_embed
import numpy as np


def test_edit():
    assert Edit("a", "b").distance() == 1
    assert Edit("a", "aa").distance() == 1
    assert Edit("scissor", "sisters").distance() == 4
    assert Edit("bonjour", "au revoir").distance() == 7

    assert len(Edit("aaa", "aaabc").compute_path()) == 3
    assert "bordeaux" in Edit("bourgogne", "bordeaux").compute_path()
    assert len(Edit("bonjour", "au revoir").compute_path()) == 8
    assert 'aaa aac abc bc' == " ".join(Edit("aaa", "bc").compute_path())


def test_embedding():
    np.random.seed(42)
    random_bits = np.ones(100, dtype=int)
    assert np.sum(Embedding("00001111", random_bits).compute()) == 0
    assert np.sum(Embedding("10001111", random_bits).compute()) == 1

    random_bits = np.random.randint(2, size=80)
    s1 = "1" * 10 + "0" * 5
    s2 = "1" * 11 + "0" * 4
    assert pair_embed(s1, s2, random_bits) == 1
    assert pair_embed("bad", "boy", random_bits) == 16
