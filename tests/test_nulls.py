"""Tests for deterministic controlled-null transformations."""

from __future__ import annotations

import numpy as np
import pytest

from market_memory_lab.nulls import (
    block_permute_returns,
    circular_shift_returns,
    derive_null_seed,
    permute_returns,
)

REFERENCE = np.array(
    [0.01, -0.02, 0.03, 0.04, -0.01, 0.02],
    dtype=float,
)


def assert_same_marginal_values(
    original: np.ndarray,
    transformed: np.ndarray,
) -> None:
    np.testing.assert_array_equal(
        np.sort(transformed),
        np.sort(original),
    )
    assert transformed.mean() == pytest.approx(
        original.mean()
    )
    assert transformed.var() == pytest.approx(
        original.var()
    )


def test_seed_derivation_is_reproducible() -> None:
    first = derive_null_seed(
        20260811,
        method="block",
        identity_a="SPY",
        identity_b="QQQ",
        parameter=21,
    )
    second = derive_null_seed(
        20260811,
        method="block",
        identity_a="SPY",
        identity_b="QQQ",
        parameter=21,
    )

    assert first == second
    assert first >= 0


def test_seed_derivation_changes_with_experiment_identity() -> None:
    baseline = derive_null_seed(
        20260811,
        method="block",
        identity_a="SPY",
        identity_b="QQQ",
        parameter=21,
    )

    alternatives = {
        derive_null_seed(
            20260811,
            method="permutation",
            identity_a="SPY",
            identity_b="QQQ",
            parameter=21,
        ),
        derive_null_seed(
            20260811,
            method="block",
            identity_a="QQQ",
            identity_b="SPY",
            parameter=21,
        ),
        derive_null_seed(
            20260811,
            method="block",
            identity_a="SPY",
            identity_b="QQQ",
            parameter=63,
        ),
    }

    assert baseline not in alternatives


@pytest.mark.parametrize(
    "invalid_seed",
    [True, -1, 1.5, "7"],
)
def test_seed_derivation_rejects_invalid_master_seed(
    invalid_seed,
) -> None:
    expected_error = (
        ValueError
        if invalid_seed == -1
        else TypeError
    )

    with pytest.raises(expected_error):
        derive_null_seed(
            invalid_seed,
            method="block",
            identity_a="A",
            identity_b="B",
        )


def test_permutation_is_deterministic_and_nonidentity() -> None:
    first = permute_returns(
        REFERENCE,
        seed=123,
    )
    second = permute_returns(
        REFERENCE,
        seed=123,
    )

    np.testing.assert_array_equal(
        first.values,
        second.values,
    )
    assert not np.array_equal(
        first.values,
        REFERENCE,
    )
    assert first.method == "return_permutation"
    assert first.seed == 123


def test_permutation_preserves_marginal_values() -> None:
    result = permute_returns(
        REFERENCE,
        seed=999,
    )

    assert_same_marginal_values(
        REFERENCE,
        result.values,
    )


def test_permutation_output_is_read_only() -> None:
    result = permute_returns(
        REFERENCE,
        seed=4,
    )

    with pytest.raises(ValueError):
        result.values[0] = 99.0


def test_circular_shift_preserves_cyclic_order() -> None:
    result = circular_shift_returns(
        REFERENCE,
        shift=2,
        seed=7,
    )

    np.testing.assert_array_equal(
        result.values,
        np.roll(REFERENCE, 2),
    )
    assert result.parameter_name == "shift"
    assert result.parameter_value == 2
    assert_same_marginal_values(
        REFERENCE,
        result.values,
    )


def test_circular_shift_normalizes_large_shift() -> None:
    result = circular_shift_returns(
        REFERENCE,
        shift=len(REFERENCE) + 2,
    )

    assert result.parameter_value == 2
    np.testing.assert_array_equal(
        result.values,
        np.roll(REFERENCE, 2),
    )


@pytest.mark.parametrize(
    "shift",
    [0, len(REFERENCE), -len(REFERENCE)],
)
def test_circular_shift_rejects_zero_equivalent_shift(
    shift: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="equivalent to zero",
    ):
        circular_shift_returns(
            REFERENCE,
            shift=shift,
        )


def test_block_permutation_preserves_values() -> None:
    result = block_permute_returns(
        REFERENCE,
        block_size=2,
        seed=123,
    )

    assert_same_marginal_values(
        REFERENCE,
        result.values,
    )
    assert not np.array_equal(
        result.values,
        REFERENCE,
    )
    assert result.parameter_name == "block_size"
    assert result.parameter_value == 2


def test_block_permutation_preserves_within_block_order() -> None:
    original = np.arange(1.0, 13.0)

    result = block_permute_returns(
        original,
        block_size=3,
        seed=88,
    )

    transformed_blocks = result.values.reshape(-1, 3)

    for block in transformed_blocks:
        assert np.all(np.diff(block) == 1.0)


def test_block_size_one_matches_element_permutation_contract() -> None:
    result = block_permute_returns(
        REFERENCE,
        block_size=1,
        seed=42,
    )

    assert_same_marginal_values(
        REFERENCE,
        result.values,
    )
    assert not np.array_equal(
        result.values,
        REFERENCE,
    )


def test_block_permutation_preserves_short_final_block() -> None:
    original = np.arange(1.0, 11.0)

    result = block_permute_returns(
        original,
        block_size=4,
        seed=55,
    )

    assert len(result.values) == len(original)
    assert_same_marginal_values(
        original,
        result.values,
    )


def test_different_seeds_can_produce_different_blocks() -> None:
    original = np.arange(1.0, 25.0)

    first = block_permute_returns(
        original,
        block_size=3,
        seed=101,
    )
    second = block_permute_returns(
        original,
        block_size=3,
        seed=202,
    )

    assert not np.array_equal(
        first.values,
        second.values,
    )


@pytest.mark.parametrize(
    "invalid_returns",
    [
        [],
        [0.01, np.nan],
        [0.01, np.inf],
        [0.01, -np.inf],
        [0.01, -1.0],
        [[0.01, 0.02]],
    ],
)
def test_transformations_reject_invalid_returns(
    invalid_returns,
) -> None:
    with pytest.raises(ValueError):
        permute_returns(
            invalid_returns,
            seed=1,
        )


@pytest.mark.parametrize(
    "invalid_block_size",
    [0, -1, True, 2.5, "5"],
)
def test_block_permutation_rejects_invalid_block_size(
    invalid_block_size,
) -> None:
    expected_error = (
        ValueError
        if invalid_block_size in {0, -1}
        else TypeError
    )

    with pytest.raises(expected_error):
        block_permute_returns(
            REFERENCE,
            block_size=invalid_block_size,
            seed=1,
        )


def test_block_permutation_rejects_single_block() -> None:
    with pytest.raises(
        ValueError,
        match="at least two blocks",
    ):
        block_permute_returns(
            REFERENCE,
            block_size=len(REFERENCE),
            seed=1,
        )


def test_transformations_do_not_mutate_original_input() -> None:
    original = REFERENCE.copy()
    preserved = original.copy()

    permute_returns(original, seed=1)
    circular_shift_returns(
        original,
        shift=2,
    )
    block_permute_returns(
        original,
        block_size=2,
        seed=1,
    )

    np.testing.assert_array_equal(
        original,
        preserved,
    )


def test_constant_returns_remain_constant() -> None:
    original = np.full(12, 0.01)

    permuted = permute_returns(
        original,
        seed=5,
    )
    shifted = circular_shift_returns(
        original,
        shift=3,
    )
    blocked = block_permute_returns(
        original,
        block_size=3,
        seed=5,
    )

    np.testing.assert_array_equal(
        permuted.values,
        original,
    )
    np.testing.assert_array_equal(
        shifted.values,
        original,
    )
    np.testing.assert_array_equal(
        blocked.values,
        original,
    )
