from edit import Edit


def test_all():
    assert Edit("a", "b").distance() == 1
    assert Edit("a", "aa").distance() == 1
    assert Edit("scissor", "sisters").distance() == 4
    assert Edit("bonjour", "au revoir").distance() == 7

    assert len(Edit("aaa", "aaabc").compute_path()) == 3
    assert "bordeaux" in Edit("bourgogne", "bordeaux").compute_path()
    assert len(Edit("bonjour", "au revoir").compute_path()) == 8
    assert 'aaa aac abc bc' == " ".join(Edit("aaa", "bc").compute_path())
