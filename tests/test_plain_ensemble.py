"""Basic tests for NeuralIL plain-ensemble models.

Covers model construction, energy/force evaluation, and the parameter
pack/unpack helpers. The pack/unpack round-trip and the import test also act
as regression guards for the jax-compatibility fix: ``unpack_params`` /
``pack_params`` (and ``ase_integration`` / the ensemble trainers) use
``jax.tree_util.tree_map``, which replaced the ``jax.tree_map`` alias that
modern jax removed -- a broken alias makes these fail.
"""

import importlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import flax.core  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from neuralil.bessel_descriptors import PowerSpectrumGenerator  # noqa: E402
from neuralil.model import NeuralIL, ResNetCore  # noqa: E402
from neuralil.plain_ensembles.model import PlainEnsemble  # noqa: E402
from neuralil.plain_ensembles.training import (  # noqa: E402
    get_n_models,
    pack_params,
    unpack_params,
)

N_MAX = 4
R_CUT = 3.0
N_TYPES = 1
EMBED_D = 1
# Distinct, decreasing widths -> projecting ResNetDense layers, so the core
# accepts the descriptor+embedding input dim directly.
CORE_WIDTHS = [32, 16]
N_ENSEMBLE = 2
MAX_NEIGHBORS = 25
N_ATOMS = 4


@pytest.fixture(scope="module")
def ensemble():
    descriptor_generator = PowerSpectrumGenerator(N_MAX, R_CUT, N_TYPES)
    core_model = ResNetCore(CORE_WIDTHS)
    individual_model = NeuralIL(
        N_TYPES,
        EMBED_D,
        R_CUT,
        descriptor_generator,
        descriptor_generator.process_some_data,
        core_model,
    )
    return PlainEnsemble(individual_model, N_ENSEMBLE)


@pytest.fixture(scope="module")
def structure():
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(1.0, R_CUT + 4.0, size=(N_ATOMS, 3)))
    types = jnp.zeros(N_ATOMS, dtype=jnp.int32)
    cell = jnp.eye(3) * 8.0
    return positions, types, cell


@pytest.fixture(scope="module")
def params(ensemble, structure):
    positions, types, cell = structure
    key = jax.random.PRNGKey(0)
    return ensemble.init(
        key,
        positions,
        types,
        cell,
        MAX_NEIGHBORS,
        method=ensemble.calc_forces,
    )


def test_training_modules_import():
    # These modules previously used the removed ``jax.tree_map``; importing
    # them must not raise on a modern jax.
    for mod in (
        "neuralil.ase_integration",
        "neuralil.plain_ensembles.training",
        "neuralil.deep_ensembles.training",
    ):
        importlib.import_module(mod)


def test_get_n_models(params):
    assert get_n_models(params) == N_ENSEMBLE


def test_unpack_params_count(params):
    individual = unpack_params(params)
    assert len(individual) == N_ENSEMBLE


def test_pack_unpack_roundtrip(params):
    # Regression guard for the jax.tree_util.tree_map fix: both helpers must
    # run and preserve the pytree structure and per-leaf shapes.
    repacked = pack_params(unpack_params(params))
    # init() may return an unfrozen dict while pack_params freezes its output;
    # normalize the freeze state before comparing the tree structure.
    assert jax.tree_util.tree_structure(
        flax.core.freeze(repacked)
    ) == jax.tree_util.tree_structure(flax.core.freeze(params))
    orig_shapes = [leaf.shape for leaf in jax.tree_util.tree_leaves(params)]
    new_shapes = [leaf.shape for leaf in jax.tree_util.tree_leaves(repacked)]
    assert new_shapes == orig_shapes


def test_model_evaluates_energy_and_forces(ensemble, params, structure):
    positions, types, cell = structure
    energy = ensemble.apply(
        params,
        positions,
        types,
        cell,
        MAX_NEIGHBORS,
        method=ensemble.calc_potential_energy,
    )
    forces = ensemble.apply(
        params,
        positions,
        types,
        cell,
        MAX_NEIGHBORS,
        method=ensemble.calc_forces,
    )
    energy = np.asarray(energy)
    forces = np.asarray(forces)
    # One energy per ensemble member; forces shaped (n_models, n_atoms, 3).
    assert energy.shape == (N_ENSEMBLE,)
    assert forces.shape == (N_ENSEMBLE, N_ATOMS, 3)
    assert np.all(np.isfinite(energy))
    assert np.all(np.isfinite(forces))
