import pytest

from plot_pareto import hypervolume, non_dominated


def test_non_dominated_keeps_only_the_front():
    points = [(1, 5), (2, 3), (3, 4), (4, 1), (2, 3)]
    assert non_dominated(points) == [(1, 5), (2, 3), (4, 1)]


def test_hypervolume_of_a_single_point():
    assert hypervolume([(1, 1)], reference=(3, 3)) == pytest.approx(4)


def test_hypervolume_of_a_staircase():
    # union of [1,3]x[2,3] and [2,3]x[1,3] = 2 + 2 - 1 overlap
    assert hypervolume([(1, 2), (2, 1)], reference=(3, 3)) == pytest.approx(3)
