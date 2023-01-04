"""
Utilities for generating sharding/partition layout specs.
"""
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze

from typing import Dict, List, Tuple, Union
from jax.sharding import PositionalSharding

ShardingScheme = Dict[str, PositionalSharding]
RecursiveShardingScheme = Dict[str, Union[ShardingScheme, "RecursiveShardingScheme"]]


def get_sharding_scheme(
    params,
    extra_keys_nonsharded: List[Tuple[str]] = [],
    num_replicas: int = 1,
) -> RecursiveShardingScheme:
    """
    Return sharding scheme for the given param dictionary.
    Shard along the axis=0 whenever possible.

    Extra keys will be included in the output and are replicated
    across all devices.
    (Cannot directly shard since dimensionality is unknown.)
    """
    assert jax.device_count() % num_replicas == 0, (
        f"Number of accelerators ({jax.device_count()})"
        + f" must be a multiple of the number of replicas ({num_replicas})"
    )
    num_devices_per_replica = jax.device_count() // num_replicas
    base_sharding = PositionalSharding(jax.devices())
    replicated_sharding = base_sharding.reshape(
        (num_devices_per_replica, num_replicas)
    ).replicate(1)

    param_shapes = jax.tree_util.tree_map(jnp.shape, params)
    param_shapes_flattened: Dict[Tuple[str], Tuple[int]]
    param_shapes_flattened = flatten_dict(param_shapes)  # type: ignore

    replicated_param_keys: Dict[Tuple[str], PositionalSharding] = {}
    for param_key, param_shape in param_shapes_flattened.items():
        param_dimensionality = len(param_shape)
        if param_shape[0] % num_devices_per_replica == 0:
            # Example: with n devices,
            # the sharding config of a 2D array (leaf) will be (n, 1).
            # So the value of the `shape` argument should be (-1, 1).
            param_sharding = replicated_sharding.reshape(
                [-1] + (param_dimensionality - 1) * [1]
            )

            replicated_param_keys[param_key] = param_sharding

        else:
            # The first axis of the param is not divisible by
            # the number of devices in each replica.
            # Replicate this param across all devices instead.
            replicated_param_keys[param_key] = replicated_sharding.replicate()

    for extra_key in extra_keys_nonsharded:
        replicated_param_keys[extra_key] = base_sharding.replicate()

    return unfreeze(unflatten_dict(replicated_param_keys))  # type: ignore
