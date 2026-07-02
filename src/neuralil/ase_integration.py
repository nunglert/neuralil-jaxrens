#!/usr/bin/env python
# Copyright 2019-2024 The NeuralIL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An ASE calculator for NeuralIL force fields, single-model or ensemble.

import copy

import jax

# Keep ASE evaluations on the CPU; do it as early as possible.
# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as onp
from ase.calculators.calculator import Calculator

from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
    get_max_number_of_neighbors,
)
from neuralil.model import NeuralILModelInfo


def pick_bucket(true_max, ladder, offset=0):
    """Smallest ladder entry that accommodates ``true_max + offset``.

    Discretising the observed neighbour count onto a fixed ladder bounds the
    number of distinct JIT compilations to ``len(ladder)`` -- the same idea as
    jaxrens' bucket manager, but without the shrink/hysteresis half (evaluation
    here is static, so there is nothing to roll back). Raises when the ladder
    is exhausted, which is user-actionable (extend ``max_neighbors_list``).
    """
    target = int(true_max) + int(offset)
    for b in ladder:
        if b >= target:
            return int(b)
    raise RuntimeError(
        f"Observed max neighbour count {int(true_max)} (+offset {offset}) "
        f"needs a bucket >= {target}, but max_neighbors_list={list(ladder)} "
        "has no entry that large. Extend it."
    )


def build_model_from_info(model_info, supercell_diag=(1, 1, 1), ensemble=True):
    """Reconstruct a NeuralIL model (single or ensemble) from its model info.

    Mirrors the training-time construction: a shared descriptor generator and
    ResNet core, optionally wrapped in the plain ensemble. ``max_neighbors`` is
    *not* baked in -- it is supplied at each evaluation.

    Args:
        model_info: A :class:`~neuralil.model.NeuralILModelInfo`, or a path to a
            pickle of one.
        supercell_diag: Diagonal supercell replication for the descriptor
            generator (for cells smaller than the cutoff sphere).
        ensemble: If True (default, the NNFF convention), wrap the model in a
            ``PlainEnsemble``; if False, return the single NeuralIL model.

    Returns:
        ``(model_info, model)``.
    """
    import pickle

    from neuralil.model import NeuralIL, NeuralILwithMorse, ResNetCore
    from neuralil.plain_ensembles.model import (
        PlainEnsemble,
        PlainEnsemblewithMorse,
    )
    from neuralil.plain_ensembles.training import get_n_models

    if not isinstance(model_info, NeuralILModelInfo):
        model_info = pickle.load(open(model_info, "rb"))

    descriptor_generator = PowerSpectrumGenerator(
        model_info.n_max,
        model_info.r_cut,
        len(model_info.sorted_elements),
        tuple(supercell_diag),
    )
    core_model = ResNetCore(model_info.core_widths)
    # An ensemble nests the wrapped NeuralIL params under "neuralil"; a single
    # model's params are flat. Look at whichever holds the NeuralIL submodules.
    top = model_info.params["params"]
    neuralil_params = top["neuralil"] if "neuralil" in top else top
    with_morse = "morse" in neuralil_params

    if with_morse:
        individual = NeuralILwithMorse(
            len(model_info.sorted_elements),
            model_info.embed_d,
            model_info.r_cut,
            descriptor_generator,
            descriptor_generator.process_some_data,
            core_model,
            morse_type="RepulsiveMorse",
        )
    else:
        individual = NeuralIL(
            len(model_info.sorted_elements),
            model_info.embed_d,
            model_info.r_cut,
            descriptor_generator,
            descriptor_generator.process_some_data,
            core_model,
        )

    if not ensemble:
        return model_info, individual

    n_models = get_n_models(model_info.params)
    wrapper = PlainEnsemblewithMorse if with_morse else PlainEnsemble
    return model_info, wrapper(individual, n_models)


class NeuralILASECalculator(Calculator):
    """ASE calculator for a NeuralIL force field -- single model *or* ensemble.

    Both model types expose the same
    ``calc_potential_energy``/``calc_forces(positions, types, cell,
    max_neighbors)`` interface. An ensemble
    (:class:`~neuralil.plain_ensembles.model.PlainEnsemble`) returns a leading
    per-model axis; this calculator detects that (the module carries an
    ``n_models`` attribute), reduces it to the mean, and reports the ensemble
    spread as an uncertainty (``energy_uncert``, ``energy_uncert_relative``,
    ``forces_uncert``). A single model is an ensemble of one -- no reduction.

    **Neighbour buckets.** ``max_neighbors`` is a static (shape-determining)
    argument of the model, so a distinct value forces a recompile. Rather than
    fix one value, pass ``max_neighbors_list`` (a ladder); each evaluation picks
    the smallest bucket that fits the structure's actual neighbour count (plus
    ``neighbor_offset`` headroom), bounding recompiles to ``len(ladder)``. A
    lone ``max_neighbors`` is just a one-entry ladder.

    **Batched evaluation.** :meth:`evaluate` scores many structures at once via
    ``vmap``, chunked by ``batch_size`` so memory stays bounded, and optionally
    sharded across local devices with ``pmap`` (``shard=True``). Structures are
    padded to a common atom count with ignored ``type=-1`` atoms, so a batch may
    mix sizes.

    Args:
        model: A ``NeuralIL`` (single) or ``PlainEnsemble`` (ensemble) module.
        model_info: The matching :class:`~neuralil.model.NeuralILModelInfo`.
        max_neighbors: A single neighbour cap (a one-entry ladder). Provide this
            or ``max_neighbors_list``.
        max_neighbors_list: Bucket ladder (ascending) for dynamic selection.
        neighbor_offset: Headroom added to the observed count before bucketing.
        n_devices: Devices to shard over in :meth:`evaluate` (defaults to all
            local devices).
    """

    implemented_properties = ("energy", "forces")
    excluded_properties = ("initial_charges", "initial_magmoms")

    def __init__(
        self,
        model,
        model_info,
        max_neighbors=None,
        max_neighbors_list=None,
        neighbor_offset=0,
        n_devices=None,
    ):
        self.calculator_results = dict()
        if not isinstance(model_info, NeuralILModelInfo):
            raise ValueError(
                "model_info must be an instance of NeuralILModelInfo"
            )
        if max_neighbors_list:
            self.ladder = tuple(sorted(int(b) for b in max_neighbors_list))
        elif max_neighbors is not None:
            self.ladder = (int(max_neighbors),)
        else:
            raise ValueError(
                "provide max_neighbors or max_neighbors_list"
            )
        self.neighbor_offset = int(neighbor_offset)
        self.max_neighbors = self.ladder[-1]  # largest bucket (compat)

        self.model = model
        self.model_info = copy.deepcopy(model_info)
        self.params = model_info.params
        # PlainEnsemble carries ``n_models``; a single NeuralIL does not.
        self.is_ensemble = hasattr(model, "n_models")
        self.symbol_map = {
            s: i for i, s in enumerate(model_info.sorted_elements)
        }
        self.n_devices = n_devices or len(jax.local_devices())
        # Compiled functions are cached per (bucket, forces?, batched?, shard?).
        self._fn_cache = {}
        super().__init__()

    @classmethod
    def from_pickle(
        cls,
        model_info,
        max_neighbors=None,
        max_neighbors_list=None,
        neighbor_offset=0,
        supercell_diag=(1, 1, 1),
        ensemble=True,
        n_devices=None,
    ):
        """Build the calculator from a pickled ``NeuralILModelInfo`` (or an info
        object). ``ensemble`` selects a ``PlainEnsemble`` (the NNFF default,
        giving uncertainties) versus a single model.
        """
        info, model = build_model_from_info(
            model_info, supercell_diag=supercell_diag, ensemble=ensemble
        )
        return cls(
            model,
            info,
            max_neighbors=max_neighbors,
            max_neighbors_list=max_neighbors_list,
            neighbor_offset=neighbor_offset,
            n_devices=n_devices,
        )

    # -- compiled-function factory ---------------------------------------
    def _apply(self, bucket, forces):
        """A single-structure ``apply`` closed over a bucket (no jit/vmap)."""
        method = (
            self.model.calc_forces
            if forces
            else self.model.calc_potential_energy
        )

        def apply(positions, types, cell):
            return self.model.apply(
                self.params, positions, types, cell, bucket, method=method
            )

        return apply

    def _single_fn(self, bucket, forces):
        key = ("single", bucket, forces)
        fn = self._fn_cache.get(key)
        if fn is None:
            fn = jax.jit(self._apply(bucket, forces))
            self._fn_cache[key] = fn
        return fn

    def _batched_fn(self, bucket, forces, sharded):
        key = ("batch", bucket, forces, sharded)
        fn = self._fn_cache.get(key)
        if fn is None:
            vmapped = jax.vmap(self._apply(bucket, forces), in_axes=(0, 0, 0))
            fn = jax.pmap(vmapped) if sharded else jax.jit(vmapped)
            self._fn_cache[key] = fn
        return fn

    # -- neighbour bucketing ---------------------------------------------
    def _bucket_for(self, positions, jtypes, cell):
        n_neighbors = get_max_number_of_neighbors(
            positions, jtypes, self.model_info.r_cut, cell
        )
        return pick_bucket(int(n_neighbors), self.ladder, self.neighbor_offset)

    def _parse(self, atoms):
        jtypes = jnp.asarray([self.symbol_map[s] for s in atoms.symbols])
        return (
            jnp.asarray(atoms.positions),
            jtypes,
            jnp.asarray(atoms.cell[...]),
        )

    # -- ASE single-structure interface ----------------------------------
    def get_potential_energy(self, atoms, *args, **kwargs):
        positions, jtypes, cell = self._parse(atoms)
        bucket = self._bucket_for(positions, jtypes, cell)
        energy = self._single_fn(bucket, False)(positions, jtypes, cell)
        if self.is_ensemble:
            value = float(energy.mean())
            uncert = float(energy.std())
            atoms.info["energy_uncert"] = uncert
            atoms.info["energy_uncert_relative"] = (
                uncert / abs(value) if value else uncert
            )
            return value
        return float(energy)

    def get_forces(self, atoms, *args, **kwargs):
        positions, jtypes, cell = self._parse(atoms)
        bucket = self._bucket_for(positions, jtypes, cell)
        forces = self._single_fn(bucket, True)(positions, jtypes, cell)
        if self.is_ensemble:
            atoms.info["forces_uncert"] = float(jnp.std(forces, axis=0).sum())
            forces = forces.mean(axis=0)
        return onp.array(forces)

    # -- batched evaluation ----------------------------------------------
    def evaluate(self, atoms_list, batch_size=None, shard=False):
        """Energies + forces (+ uncertainties) for many structures at once.

        Runs a ``vmap`` over the batch, chunked to at most ``batch_size``
        structures so peak memory is bounded, and -- when ``shard`` -- splits
        each chunk across ``n_devices`` with ``pmap``. Structures are padded to
        a common atom count with ignored ``type=-1`` atoms and a single bucket
        (the max over the batch) is used, so a batch may mix sizes/densities.

        Args:
            atoms_list: Structures to evaluate.
            batch_size: Max structures per ``vmap`` (default: all of them).
            shard: Split each chunk across local devices with ``pmap``.

        Returns:
            A list of per-structure dicts with ``energy`` and ``forces`` (and,
            for an ensemble, ``energy_uncert``/``energy_uncert_relative`` and
            ``forces_uncert``). Forces are trimmed back to each structure's own
            atom count.
        """
        n_struct = len(atoms_list)
        if n_struct == 0:
            return []
        n_atoms = max(len(a) for a in atoms_list)

        P = onp.zeros((n_struct, n_atoms, 3))
        T = -onp.ones((n_struct, n_atoms), dtype=int)  # -1 = ignored padding
        C = onp.zeros((n_struct, 3, 3))
        counts = onp.zeros(n_struct, dtype=int)
        true_max = 0
        for i, a in enumerate(atoms_list):
            na = len(a)
            counts[i] = na
            P[i, :na] = a.positions
            T[i, :na] = [self.symbol_map[s] for s in a.symbols]
            C[i] = a.cell[...]
            n_nb = get_max_number_of_neighbors(
                jnp.asarray(a.positions),
                jnp.asarray(T[i, :na]),
                self.model_info.r_cut,
                jnp.asarray(a.cell[...]),
            )
            true_max = max(true_max, int(n_nb))
        bucket = pick_bucket(true_max, self.ladder, self.neighbor_offset)

        n_dev = self.n_devices if shard else 1
        batch_size = batch_size or n_struct

        e_chunks, f_chunks = [], []
        for start in range(0, n_struct, batch_size):
            p = P[start : start + batch_size]
            t = T[start : start + batch_size]
            c = C[start : start + batch_size]
            b = p.shape[0]
            if n_dev > 1:
                pad = (-b) % n_dev
                if pad:  # replicate the last structure to fill the device grid
                    p = onp.concatenate([p, onp.repeat(p[-1:], pad, 0)])
                    t = onp.concatenate([t, onp.repeat(t[-1:], pad, 0)])
                    c = onp.concatenate([c, onp.repeat(c[-1:], pad, 0)])
                per = (b + pad) // n_dev
                rs = lambda x: jnp.asarray(x).reshape(n_dev, per, *x.shape[1:])
                e = self._batched_fn(bucket, False, True)(rs(p), rs(t), rs(c))
                f = self._batched_fn(bucket, True, True)(rs(p), rs(t), rs(c))
                e = e.reshape(-1, *e.shape[2:])[:b]
                f = f.reshape(-1, *f.shape[2:])[:b]
            else:
                p, t, c = jnp.asarray(p), jnp.asarray(t), jnp.asarray(c)
                e = self._batched_fn(bucket, False, False)(p, t, c)
                f = self._batched_fn(bucket, True, False)(p, t, c)
            e_chunks.append(e)
            f_chunks.append(f)

        E = onp.asarray(jnp.concatenate(e_chunks, axis=0))
        F = onp.asarray(jnp.concatenate(f_chunks, axis=0))

        results = []
        for i in range(n_struct):
            na = counts[i]
            if self.is_ensemble:
                e_i, f_i = E[i], F[i]  # (n_models,), (n_models, n_atoms, 3)
                value = float(e_i.mean())
                uncert = float(e_i.std())
                results.append(
                    {
                        "energy": value,
                        "forces": f_i.mean(axis=0)[:na],
                        "energy_uncert": uncert,
                        "energy_uncert_relative": (
                            uncert / abs(value) if value else uncert
                        ),
                        "forces_uncert": float(
                            onp.std(f_i, axis=0)[:na].sum()
                        ),
                    }
                )
            else:
                results.append(
                    {"energy": float(E[i]), "forces": F[i][:na]}
                )
        return results
