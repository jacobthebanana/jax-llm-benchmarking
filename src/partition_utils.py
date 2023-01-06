"""
Utilities for sharding/partitioning LLM parameters.
"""
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze
import chex

from typing import Dict, List, Tuple, Union
from jax.sharding import PositionalSharding

# Typing
from jaxlib.xla_extension import Device

ShardingScheme = Dict[str, PositionalSharding]
RecursiveShardingScheme = Dict[str, Union[ShardingScheme, "RecursiveShardingScheme"]]


def get_sharding_scheme(
    params,
    extra_keys_nonsharded: List[Tuple[str]] = [],
    num_replicas: int = 1,
) -> RecursiveShardingScheme:
    """
    Return the "full" sharding scheme for the given param dictionary.
    Shard along the axis=0 whenever possible. This sharding scheme
    includes all devices in the pod slice, possibly including ones
    that this CPU host cannot address.

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


def device_put_leaf(
    leaf_array: jnp.ndarray, global_leaf_sharding: PositionalSharding
) -> jnp.ndarray:
    """
    Given a leaf array and its "global" sharding scheme
    (might include non-addressable devices), place shards of the leaf
    array belonging to this host
    (specifically, accelerators addressable by the current CPU host.)
    to the appropriate accelerators.

    Note that "slice" refers to the tuples that index
    a high-dimensional jax.numpy array.

    Params:
      leaf_array: "leaf" of the param tree, including values that do
      not belong to accelerator devices of the current CPU host.

      global_leaf_sharding: PositionalSharding
      (might include non-local accelerator devices.)

    Returns:
      ShardedDeviceArray
    """
    local_array_buffers = []
    slices_by_device = global_leaf_sharding.addressable_devices_indices_map(
        leaf_array.shape
    )

    for device in jax.local_devices():
        shard_slice = slices_by_device[device]
        local_array = jax.device_put(leaf_array[shard_slice], device)
        local_array_buffers.append(local_array)

    sharded_leaf_array = jax.make_array_from_single_device_arrays(
        leaf_array.shape, global_leaf_sharding, local_array_buffers
    )

    chex.assert_equal_shape([leaf_array, sharded_leaf_array])
    return sharded_leaf_array
