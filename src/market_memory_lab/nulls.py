"""Deterministic return transformations for controlled null experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NullTransformation:
    """One transformed return sequence with reproducible metadata."""

    values: FloatArray
    method: str
    seed: int
    original_length: int
    parameter_name: str | None = None
    parameter_value: int | None = None

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("method must not be empty")

        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

        if self.original_length <= 0:
            raise ValueError("original_length must be positive")

        if self.values.ndim != 1:
            raise ValueError("values must be one-dimensional")

        if len(self.values) != self.original_length:
            raise ValueError(
                "values length must equal original_length"
            )

        if not np.isfinite(self.values).all():
            raise ValueError(
                "values must contain only finite returns"
            )

        if (
            self.parameter_name is None
            and self.parameter_value is not None
        ):
            raise ValueError(
                "parameter_value requires parameter_name"
            )

        if (
            self.parameter_name is not None
            and not self.parameter_name.strip()
        ):
            raise ValueError(
                "parameter_name must not be blank"
            )

        self.values.setflags(write=False)


def derive_null_seed(
    master_seed: int,
    *,
    method: str,
    identity_a: str,
    identity_b: str,
    parameter: int | None = None,
) -> int:
    """Derive a stable unsigned seed from explicit experiment identity."""
    if isinstance(master_seed, bool) or not isinstance(
        master_seed,
        int,
    ):
        raise TypeError("master_seed must be an integer")

    if master_seed < 0:
        raise ValueError("master_seed must be nonnegative")

    required_text = {
        "method": method,
        "identity_a": identity_a,
        "identity_b": identity_b,
    }

    for field_name, value in required_text.items():
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        if not value.strip():
            raise ValueError(
                f"{field_name} must not be empty"
            )

    payload = "|".join(
        (
            str(master_seed),
            method.strip(),
            identity_a.strip(),
            identity_b.strip(),
            "" if parameter is None else str(parameter),
        )
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def _validated_returns(
    returns: NDArray[np.floating] | list[float] | tuple[float, ...],
) -> FloatArray:
    values = np.asarray(returns, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")

    if values.size == 0:
        raise ValueError("returns must not be empty")

    if not np.isfinite(values).all():
        raise ValueError(
            "returns must contain only finite values"
        )

    if (values <= -1.0).any():
        raise ValueError(
            "simple returns must be greater than -1"
        )

    return values.copy()


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be nonnegative")

    return seed


def _force_nonidentity_order(
    order: NDArray[np.integer],
) -> NDArray[np.integer]:
    identity = np.arange(len(order))

    if np.array_equal(order, identity):
        return np.roll(order, 1)

    return order


def permute_returns(
    returns: NDArray[np.floating] | list[float] | tuple[float, ...],
    *,
    seed: int,
) -> NullTransformation:
    """Permute individual returns while preserving their multiset."""
    values = _validated_returns(returns)
    selected_seed = _validated_seed(seed)

    if len(values) < 2:
        raise ValueError(
            "permutation requires at least two returns"
        )

    generator = np.random.default_rng(selected_seed)
    order = _force_nonidentity_order(
        generator.permutation(len(values))
    )
    transformed = values[order].copy()

    return NullTransformation(
        values=transformed,
        method="return_permutation",
        seed=selected_seed,
        original_length=len(values),
    )


def circular_shift_returns(
    returns: NDArray[np.floating] | list[float] | tuple[float, ...],
    *,
    shift: int,
    seed: int = 0,
) -> NullTransformation:
    """Circularly shift a complete return sequence."""
    values = _validated_returns(returns)
    selected_seed = _validated_seed(seed)

    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("shift must be an integer")

    if len(values) < 2:
        raise ValueError(
            "circular shift requires at least two returns"
        )

    normalized_shift = shift % len(values)

    if normalized_shift == 0:
        raise ValueError(
            "shift must not be equivalent to zero"
        )

    transformed = np.roll(
        values,
        normalized_shift,
    ).copy()

    return NullTransformation(
        values=transformed,
        method="circular_shift",
        seed=selected_seed,
        original_length=len(values),
        parameter_name="shift",
        parameter_value=normalized_shift,
    )


def block_permute_returns(
    returns: NDArray[np.floating] | list[float] | tuple[float, ...],
    *,
    block_size: int,
    seed: int,
) -> NullTransformation:
    """Permute blocks while preserving order inside every block."""
    values = _validated_returns(returns)
    selected_seed = _validated_seed(seed)

    if isinstance(block_size, bool) or not isinstance(
        block_size,
        int,
    ):
        raise TypeError("block_size must be an integer")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    block_starts = range(0, len(values), block_size)
    blocks = [
        values[start : start + block_size].copy()
        for start in block_starts
    ]

    if len(blocks) < 2:
        raise ValueError(
            "block permutation requires at least two blocks"
        )

    generator = np.random.default_rng(selected_seed)
    order = _force_nonidentity_order(
        generator.permutation(len(blocks))
    )

    transformed = np.concatenate(
        [blocks[int(index)] for index in order]
    ).astype(np.float64, copy=False)

    if len(transformed) != len(values):
        raise AssertionError(
            "block permutation changed sequence length"
        )

    return NullTransformation(
        values=transformed.copy(),
        method="block_permutation",
        seed=selected_seed,
        original_length=len(values),
        parameter_name="block_size",
        parameter_value=block_size,
    )
