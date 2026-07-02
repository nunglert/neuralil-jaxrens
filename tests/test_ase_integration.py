"""Tests for the NeuralIL ASE calculator (``neuralil.ase_integration``).

Covers, against real (tiny) models:

* ``pick_bucket`` ladder logic (pure);
* the single class serving both a single model and an ensemble (uncertainty
  only for the latter);
* ``from_pickle`` reconstruction;
* neighbour-bucket selection and ladder exhaustion;
* batched ``evaluate`` -- consistency with the per-structure path, ``batch_size``
  chunking, mixed atom counts (padding), and sharding correctness.

Two CPU devices are faked (via ``XLA_FLAGS``) *before* jax imports so the
sharded path can exercise ``pmap`` in-process. If jax was already initialised
with one device, the sharded test still checks correctness (it just falls back
to the vmap path).
"""

import os

# Must be set before jax initialises to take effect.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2"
)

import datetime  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from ase import Atoms  # noqa: E402

from neuralil.ase_integration import (  # noqa: E402
    NeuralILASECalculator,
    build_model_from_info,
    pick_bucket,
)
from neuralil.model import NeuralIL, NeuralILModelInfo, ResNetCore  # noqa: E402
from neuralil.plain_ensembles.model import PlainEnsemble  # noqa: E402

N_MAX = 4
R_CUT = 3.0
N_TYPES = 1
EMBED_D = 1
CORE_WIDTHS = [32, 16]
N_ENSEMBLE = 2
MAX_NEIGHBORS = 25
ELEMENTS = ["Cu"]


# ---------------------------------------------------------------------------
# pick_bucket -- pure logic
# ---------------------------------------------------------------------------
def test_pick_bucket_smallest_fitting_entry():
    ladder = (20, 30, 40)
    assert pick_bucket(5, ladder) == 20
    assert pick_bucket(20, ladder) == 20
    assert pick_bucket(21, ladder) == 30
    assert pick_bucket(40, ladder) == 40


def test_pick_bucket_offset_headroom():
    ladder = (20, 30, 40)
    # 19 + offset 2 = 21 -> needs the next bucket up.
    assert pick_bucket(19, ladder, offset=2) == 30


def test_pick_bucket_exhausted_raises():
    with pytest.raises(RuntimeError, match="Extend it"):
        pick_bucket(100, (20, 30, 40))


# ---------------------------------------------------------------------------
# Fixtures: tiny real models + a NeuralILModelInfo
# ---------------------------------------------------------------------------
def _descriptor_generator():
    from neuralil.bessel_descriptors import PowerSpectrumGenerator

    return PowerSpectrumGenerator(N_MAX, R_CUT, N_TYPES)


def _single_model():
    dg = _descriptor_generator()
    return NeuralIL(
        N_TYPES, EMBED_D, R_CUT, dg, dg.process_some_data, ResNetCore(CORE_WIDTHS)
    )


def _model_info(params, model_name):
    return NeuralILModelInfo(
        model_name=model_name,
        model_version="test",
        timestamp=datetime.datetime(2020, 1, 1),
        r_cut=R_CUT,
        n_max=N_MAX,
        sorted_elements=ELEMENTS,
        embed_d=EMBED_D,
        core_widths=CORE_WIDTHS,
        constructor_kwargs={},
        random_seed=0,
        params=params,
        specific_info=None,
    )


def _atoms(n_atoms, seed=0, box=8.0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(1.0, box - 1.0, size=(n_atoms, 3))
    at = Atoms(
        "Cu" + str(n_atoms),
        positions=positions,
        cell=np.eye(3) * box,
        pbc=True,
    )
    return at


@pytest.fixture(scope="module")
def ensemble_calc():
    model = PlainEnsemble(_single_model(), N_ENSEMBLE)
    at = _atoms(4)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.asarray(at.positions),
        jnp.zeros(len(at), dtype=jnp.int32),
        jnp.asarray(at.cell[...]),
        MAX_NEIGHBORS,
        method=model.calc_forces,
    )
    info = _model_info(params, "PlainEnsemble")
    calc = NeuralILASECalculator(model, info, max_neighbors=MAX_NEIGHBORS)
    return calc, model, info


@pytest.fixture(scope="module")
def single_calc():
    model = _single_model()
    at = _atoms(4)
    params = model.init(
        jax.random.PRNGKey(1),
        jnp.asarray(at.positions),
        jnp.zeros(len(at), dtype=jnp.int32),
        jnp.asarray(at.cell[...]),
        MAX_NEIGHBORS,
        method=model.calc_forces,
    )
    info = _model_info(params, "NeuralIL")
    calc = NeuralILASECalculator(model, info, max_neighbors=MAX_NEIGHBORS)
    return calc, model, info


# ---------------------------------------------------------------------------
# Ensemble vs single model behaviour
# ---------------------------------------------------------------------------
def test_ensemble_energy_is_member_mean_with_uncertainty(ensemble_calc):
    calc, model, info = ensemble_calc
    at = _atoms(4)
    energy = calc.get_potential_energy(at)

    member_energies = np.asarray(
        model.apply(
            info.params,
            jnp.asarray(at.positions),
            jnp.zeros(len(at), dtype=jnp.int32),
            jnp.asarray(at.cell[...]),
            MAX_NEIGHBORS,
            method=model.calc_potential_energy,
        )
    )
    assert np.isfinite(energy)
    assert np.isclose(energy, member_energies.mean(), rtol=1e-5)
    # Uncertainty is the ensemble spread and is recorded.
    assert np.isclose(
        at.info["energy_uncert"], member_energies.std(), rtol=1e-5
    )
    assert "energy_uncert_relative" in at.info


def test_ensemble_forces_shape_and_uncertainty(ensemble_calc):
    calc, _, _ = ensemble_calc
    at = _atoms(4)
    forces = calc.get_forces(at)
    assert forces.shape == (len(at), 3)
    assert np.all(np.isfinite(forces))
    assert "forces_uncert" in at.info


def test_single_model_has_no_uncertainty(single_calc):
    calc, _, _ = single_calc
    at = _atoms(4)
    energy = calc.get_potential_energy(at)
    forces = calc.get_forces(at)
    assert np.isfinite(energy)
    assert forces.shape == (len(at), 3)
    assert calc.is_ensemble is False
    assert "energy_uncert" not in at.info
    assert "forces_uncert" not in at.info


# ---------------------------------------------------------------------------
# from_pickle reconstruction
# ---------------------------------------------------------------------------
def test_from_pickle_reconstructs_and_matches(ensemble_calc, tmp_path):
    calc, _, info = ensemble_calc
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(info, f)

    rebuilt = NeuralILASECalculator.from_pickle(
        str(path), max_neighbors=MAX_NEIGHBORS, ensemble=True
    )
    at = _atoms(4)
    assert np.isclose(
        rebuilt.get_potential_energy(at),
        calc.get_potential_energy(_atoms(4)),
        rtol=1e-5,
    )


def test_build_model_from_info_single_vs_ensemble(single_calc):
    _, _, info = single_calc
    _, model = build_model_from_info(info, ensemble=False)
    assert not hasattr(model, "n_models")


# ---------------------------------------------------------------------------
# Neighbour buckets
# ---------------------------------------------------------------------------
def test_ladder_selects_bucket_and_evaluates():
    model = PlainEnsemble(_single_model(), N_ENSEMBLE)
    at = _atoms(4)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.asarray(at.positions),
        jnp.zeros(len(at), dtype=jnp.int32),
        jnp.asarray(at.cell[...]),
        40,
        method=model.calc_forces,
    )
    info = _model_info(params, "PlainEnsemble")
    calc = NeuralILASECalculator(
        model, info, max_neighbors_list=[20, 30, 40], neighbor_offset=1
    )
    assert calc.ladder == (20, 30, 40)
    assert np.isfinite(calc.get_potential_energy(_atoms(4)))


def test_ladder_exhausted_raises(ensemble_calc):
    _, model, info = ensemble_calc
    # A dense cluster has many neighbours; a tiny ladder cannot hold them.
    calc = NeuralILASECalculator(model, info, max_neighbors_list=[1])
    dense = Atoms(
        "Cu8",
        positions=np.random.default_rng(3).uniform(0, 1.5, size=(8, 3)),
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    with pytest.raises(RuntimeError, match="Extend it"):
        calc.get_potential_energy(dense)


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------
def test_evaluate_matches_single_path(ensemble_calc):
    calc, _, _ = ensemble_calc
    structures = [_atoms(4, seed=i) for i in range(3)]
    results = calc.evaluate(structures)

    assert len(results) == 3
    for at, res in zip(structures, results):
        expected = calc.get_potential_energy(at)
        assert np.isclose(res["energy"], expected, rtol=1e-5)
        assert res["forces"].shape == (len(at), 3)
        assert "energy_uncert" in res and "forces_uncert" in res


def test_evaluate_batch_size_is_consistent(ensemble_calc):
    calc, _, _ = ensemble_calc
    structures = [_atoms(4, seed=i) for i in range(5)]
    full = calc.evaluate(structures, batch_size=None)
    chunked = calc.evaluate(structures, batch_size=2)
    for a, b in zip(full, chunked):
        assert np.isclose(a["energy"], b["energy"], rtol=1e-5)


def test_evaluate_mixed_sizes_trims_forces(ensemble_calc):
    calc, _, _ = ensemble_calc
    small, big = _atoms(4, seed=1), _atoms(6, seed=2)
    results = calc.evaluate([small, big])
    assert results[0]["forces"].shape == (4, 3)
    assert results[1]["forces"].shape == (6, 3)
    # The padded structure's energy matches its standalone evaluation.
    assert np.isclose(
        results[0]["energy"], calc.get_potential_energy(small), rtol=1e-5
    )


def test_evaluate_sharded_matches_unsharded(ensemble_calc):
    """The shard path (pmap when >1 device) yields the same numbers."""
    calc, _, _ = ensemble_calc
    structures = [_atoms(4, seed=i) for i in range(4)]
    plain = calc.evaluate(structures, shard=False)
    sharded = calc.evaluate(structures, shard=True)
    for a, b in zip(plain, sharded):
        assert np.isclose(a["energy"], b["energy"], rtol=1e-5)
        assert np.allclose(a["forces"], b["forces"], rtol=1e-5, atol=1e-6)
