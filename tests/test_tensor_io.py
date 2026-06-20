import numpy as np

from qsfl.utils.tensor_io import pack_state, unpack_state


def test_pack_unpack_roundtrip():
    state = {
        "layer2.weight": np.random.randn(4, 3).astype(np.float32),
        "layer1.bias": np.random.randn(8).astype(np.float32),
        "embed": np.arange(10, dtype=np.float64),
    }
    blob = pack_state(state)
    restored = unpack_state(blob)
    assert set(restored) == set(state)
    for k in state:
        assert restored[k].dtype == state[k].dtype
        np.testing.assert_array_equal(restored[k], state[k])


def test_pack_is_deterministic_and_order_independent():
    a = {"w": np.ones((2, 2), np.float32), "b": np.zeros(2, np.float32)}
    b = {"b": np.zeros(2, np.float32), "w": np.ones((2, 2), np.float32)}
    assert pack_state(a) == pack_state(b)  # sorted by name -> identical bytes
