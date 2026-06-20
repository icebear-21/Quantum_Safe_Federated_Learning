import numpy as np

from qsfl.federated.secure_agg import (
    add_states,
    generate_pairwise_mask,
    scale_state,
    sum_states,
    weighted_average,
)


def test_pairwise_masks_cancel():
    participants = [0, 1, 2, 3]
    template = {"w": np.zeros((3, 4)), "b": np.zeros(5)}
    masks = [generate_pairwise_mask(i, participants, template, seed=42) for i in participants]
    total = sum_states(masks)
    for v in total.values():
        np.testing.assert_allclose(v, 0.0, atol=1e-9)


def test_masked_aggregation_matches_weighted_average():
    rng = np.random.default_rng(0)
    participants = [0, 1, 2]
    deltas = [{"w": rng.standard_normal((2, 2))} for _ in participants]
    weights = [10.0, 20.0, 30.0]
    template = {"w": np.zeros((2, 2))}

    masked = [
        add_states(scale_state(deltas[i], weights[i]), generate_pairwise_mask(i, participants, template, 7))
        for i in participants
    ]
    agg_masked = scale_state(sum_states(masked), 1.0 / sum(weights))
    agg_plain = weighted_average(deltas, weights)
    np.testing.assert_allclose(agg_masked["w"], agg_plain["w"], atol=1e-9)


def test_weighted_average_simple():
    a = {"w": np.array([0.0, 0.0])}
    b = {"w": np.array([2.0, 4.0])}
    out = weighted_average([a, b], [1.0, 3.0])  # (0*1 + 2*3)/4 = 1.5 ; (0+12)/4 = 3
    np.testing.assert_allclose(out["w"], [1.5, 3.0])
