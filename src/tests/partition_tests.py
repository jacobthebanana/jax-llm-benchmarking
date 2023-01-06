import unittest

import os

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.experimental.multihost_utils import process_allgather
from transformers import FlaxOPTModel, AutoTokenizer

from ..partition_utils import get_sharding_scheme, device_put_leaf

HF_MODEL_NAME = os.environ.get("TEST_HF_MODEL", "facebook/opt-350m")
BLOCK_SIZE = int(os.environ.get("TEST_BLOCK_SIZE", "128"))


class StaticPartitioningTests(unittest.TestCase):
    """
    Test cases for the partition utilities.
    """

    @classmethod
    def setUpClass(cls):
        super(StaticPartitioningTests, cls).setUpClass()

        jax.distributed.initialize()

    def setUp(self):
        print(__name__)
        self.example_array = jnp.arange(128).reshape((16, 8))
        self.example_params = {"example": self.example_array}

        self.sharding_scheme = get_sharding_scheme(self.example_params)

    def test_sharded_params_placement(self):
        sharded_params = tree_map(
            device_put_leaf, self.example_params, self.sharding_scheme
        )

        sharded_params_cpu = process_allgather(sharded_params)
        assert isinstance(sharded_params_cpu, dict)
        assert jnp.allclose(sharded_params_cpu["example"], self.example_array)


class LMPartitioningTests(unittest.TestCase):
    """
    Test cases for the partition utilities on HuggingFace LMs.
    """

    @classmethod
    def setUpClass(cls):
        super(LMPartitioningTests, cls).setUpClass()

        cls.model, cls.params = FlaxOPTModel.from_pretrained(
            HF_MODEL_NAME, _do_init=False
        )  # type: ignore

    def setUp(self):
        self.sharding_scheme = get_sharding_scheme(LMPartitioningTests.params)

        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.example_batch = self.tokenizer(
            ["Vector Institute"],
            max_length=128,
            padding="max_length",
            return_tensors="jax",
        )

    def test_sharded_forward(self):
        sharded_params = tree_map(
            device_put_leaf, LMPartitioningTests.params, self.sharding_scheme
        )
        model_fn = jax.jit(LMPartitioningTests.model.__call__)

        with jax.spmd_mode("allow_all"):
            example_output = model_fn(**self.example_batch, params=sharded_params)

        print(tree_map(jnp.shape, process_allgather(example_output)))
