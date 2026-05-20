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

"""Soft-core NeuralIL variants.

A `SoftCoreNeuralIL` augments the standard `NeuralIL` model with a fixed
(non-trainable) repulsive Morse term that smoothly switches off between
`r_core_switch` and `r_core_cut`. This guarantees a strong repulsion at
close inter-atomic distances irrespective of the NN training distribution,
which is useful for samplers (e.g. nested sampling) that may explore
unphysical close-contact configurations far outside the training set.
"""

from typing import ClassVar

import flax.linen
import jax
import jax.numpy as jnp

from neuralil import __version__ as package_version
from neuralil.model import NeuralIL, smooth_cutoff
from neuralil.plain_ensembles.model import PlainEnsemble


class FixedRepulsiveMorse(flax.linen.Module):
    """Fixed-parameter repulsive Morse potential with a smooth cutoff.

    The potential function is

        phi(r) = d0 * exp(-2 * a0 * (r - b0))

    multiplied by `smooth_cutoff(r, r_switch, r_cut)`. All parameters are
    hyperparameters (no `self.param(...)` calls); the params tree is empty.

    The interface mirrors `RepulsiveMorseModel.calc_atomic_energies` so it
    drops into the same plumbing used by `NeuralILwithMorse.calc_morse_energies`.

    Args:
        a0: Fixed steepness parameter.
        b0: Fixed equilibrium offset.
        d0: Fixed prefactor.
        r_cut: Cutoff radius (potential is exactly zero beyond this).
        r_switch: Switching radius (`< r_cut`); potential starts differing
            from its bare form here.
    """

    a0: float
    b0: float
    d0: float
    r_cut: float
    r_switch: float

    def calc_atomic_energies(self, radii, probe_types, source_types):
        """Compute the soft-core contribution to the per-atom energy.

        Args:
            radii: The (n_probe, n_source) matrix of inter-atomic distances.
            probe_types: The atom types of the "probe" atoms.
            source_types: The atom types of the "source" atoms.

        Returns:
            The n_probe contributions to the energy.
        """
        # Push padded-atom rows past the cutoff so they contribute nothing.
        mask = jnp.logical_or(
            probe_types[:, jnp.newaxis] < 0,
            source_types[jnp.newaxis, :] < 0,
        )
        radii = radii + 2.0 * mask * self.r_cut

        phi = self.d0 * jnp.exp(-2.0 * self.a0 * (radii - self.b0))
        cutoffs = smooth_cutoff(radii, self.r_switch, self.r_cut)
        contributions = phi * cutoffs
        # Zero out the self-interaction (radii exactly zero on the diagonal).
        contributions *= jnp.logical_not(jnp.isclose(0.0, radii))
        return 0.5 * contributions.sum(axis=1)


class SoftCoreNeuralIL(NeuralIL):
    """NeuralIL with a fixed repulsive Morse soft core.

    The flax params tree is identical to that of `NeuralIL` (the soft-core
    carries no trainable parameters). The soft-core contribution is added
    to the NN atomic energies in `calc_atomic_energies`.

    Note: `calc_some_atomic_energies` (inherited from `NeuralIL`) is *not*
    overridden and therefore does not include the soft-core. Callers that
    rely on partial evaluations and also want the soft repulsion should
    use the full-population energy path.

    Args:
        a0, b0, d0: Fixed Morse parameters of the soft core.
        r_core_cut: Cutoff radius of the soft core.
        r_core_switch: Switching radius of the soft core (`< r_core_cut`).
    """

    a0: float = 1.0
    b0: float = 3.0
    d0: float = 1.0
    r_core_cut: float = 1.25
    r_core_switch: float = 0.75
    model_name: ClassVar[str] = "NeuralIL+SoftCore"
    model_version: ClassVar[str] = "0.1"
    neuralil_version: ClassVar[str] = package_version

    def setup(self):
        super().setup()
        self.soft_core = FixedRepulsiveMorse(
            self.a0,
            self.b0,
            self.d0,
            self.r_core_cut,
            self.r_core_switch,
        )

    def calc_morse_energies(self, positions, types, cell):
        """Per-atom soft-core (fixed Morse) contributions."""
        _, radii, all_types = self.descriptor_generator.center_at_atoms(
            positions, types, cell
        )
        morse_contributions = self.soft_core.calc_atomic_energies(
            radii, types, all_types
        )
        return (types >= 0) * morse_contributions

    def calc_atomic_energies(self, positions, types, cell, max_neighbors):
        descriptors = self.descriptor_generator(
            positions, types, cell, max_neighbors
        )
        nn_contributions = self.calc_atomic_energies_from_descriptors(
            descriptors, types
        )
        morse_contributions = self.calc_morse_energies(positions, types, cell)
        return (types >= 0) * (nn_contributions + morse_contributions)


class SoftCorePlainEnsemble(PlainEnsemble):
    """Ensemble of N SoftCoreNeuralIL models.

    The vmap setup is inherited from `PlainEnsemble`: only the NN
    descriptor head is vmapped over the ensemble axis. The soft-core
    contribution is parameter-free and is therefore evaluated *once* and
    broadcast across the ensemble axis (avoiding a redundant N-way vmap of
    an identical computation).
    """

    neuralil: SoftCoreNeuralIL
    model_name: ClassVar[str] = "PlainEnsemble+SoftCore"
    model_version: ClassVar[str] = "0.1"
    neuralil_version: ClassVar[str] = package_version

    def calc_atomic_energies(self, positions, types, cell, max_neighbors):
        descriptors = self.neuralil.descriptor_generator(
            positions, types, cell, max_neighbors
        )
        nn_contributions = self.calc_atomic_energies_from_descriptors(
            self.neuralil, descriptors, types
        )
        morse_contributions = self.neuralil.calc_morse_energies(
            positions, types, cell
        )
        return (types >= 0)[jnp.newaxis, :] * (
            nn_contributions + morse_contributions[jnp.newaxis, :]
        )
