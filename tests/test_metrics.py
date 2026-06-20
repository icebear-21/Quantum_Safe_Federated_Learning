import numpy as np
import pytest

from qsfl.utils.metrics import accuracy, bit_error_rate, normalized_correlation


def test_nc_identical_is_one():
    a = np.array([1, 0, 1, 1, 0, 1])
    assert normalized_correlation(a, a, binary=True) == pytest.approx(1.0)


def test_nc_complement_is_minus_one():
    a = np.array([1, 0, 1, 0])
    b = 1 - a
    assert normalized_correlation(a, b, binary=True) == pytest.approx(-1.0)


def test_nc_zero_vector():
    assert normalized_correlation([0, 0, 0], [1, 2, 3]) == 0.0


def test_ber_bounds_and_value():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 1])
    assert bit_error_rate(a, b) == pytest.approx(0.5)
    assert bit_error_rate(a, a) == 0.0


def test_accuracy():
    assert accuracy([1, 2, 3, 4], [1, 2, 0, 4]) == pytest.approx(0.75)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        normalized_correlation([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        bit_error_rate([1], [1, 0])
