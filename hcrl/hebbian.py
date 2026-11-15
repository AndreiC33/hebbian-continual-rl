"""Hebbian network with evolved per-synapse ABCD plasticity rules (JAX).

Follows the idea of Najarro & Risi (NeurIPS 2020): the network's weights are
NOT optimized directly. They start random at the beginning of every episode and
change at every time step according to a local Hebbian rule

    dw_ij = eta_ij * (A_ij * o_i * o_j + B_ij * o_j + C_ij * o_i + D_ij)

where o_j is the pre-synaptic activity and o_i the post-synaptic activity.
Only the rule coefficients (A, B, C, D, eta) for every synapse are evolved.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

N_RULE_COEFFS = 5  # A, B, C, D, eta


class HebbianNet(NamedTuple):
    """Static description of the network architecture."""

    layer_sizes: Sequence[int]  # e.g. (4, 32, 32, 2)

    @property
    def weight_shapes(self) -> list[tuple[int, int]]:
        return [(n_in, n_out) for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    @property
    def num_synapses(self) -> int:
        return sum(a * b for a, b in self.weight_shapes)

    @property
    def num_rule_params(self) -> int:
        """Length of the flat vector of evolved rule coefficients."""
        return N_RULE_COEFFS * self.num_synapses

    # ------------------------------------------------------------------ #
    def unflatten_rules(self, flat: jax.Array) -> list[jax.Array]:
        """Flat genome -> list of per-layer rule tensors of shape (5, n_in, n_out)."""
        rules, i = [], 0
        for n_in, n_out in self.weight_shapes:
            size = N_RULE_COEFFS * n_in * n_out
            rules.append(flat[i : i + size].reshape(N_RULE_COEFFS, n_in, n_out))
            i += size
        return rules

    def init_weights(self, key: jax.Array, scale: float = 0.1) -> list[jax.Array]:
        """Fresh random weights, U[-scale, scale], drawn at the start of each lifetime."""
        keys = jax.random.split(key, len(self.weight_shapes))
        return [jax.random.uniform(k, s, minval=-scale, maxval=scale) for k, s in zip(keys, self.weight_shapes)]

    def forward(self, weights: list[jax.Array], obs: jax.Array) -> tuple[jax.Array, list[jax.Array]]:
        """Returns the output and the activations of every layer (needed by the rule)."""
        acts = [jnp.tanh(obs)]
        x = acts[0]
        for w in weights:
            x = jnp.tanh(x @ w)
            acts.append(x)
        return x, acts

    def hebbian_update(
        self,
        weights: list[jax.Array],
        rules: list[jax.Array],
        acts: list[jax.Array],
        normalize: bool = True,
    ) -> list[jax.Array]:
        """Apply the ABCD rule to every synapse using only local pre/post activity."""
        new_weights = []
        for w, r, pre, post in zip(weights, rules, acts[:-1], acts[1:]):
            A, B, C, D, eta = r
            pre_ = pre[:, None]   # o_j, shape (n_in, 1)
            post_ = post[None, :]  # o_i, shape (1, n_out)
            dw = eta * (A * pre_ * post_ + B * pre_ + C * post_ + D)
            w = w + dw
            if normalize:
                # Keeps the self-organizing dynamics bounded (Hebbian learning alone can run away).
                w = w / (jnp.max(jnp.abs(w)) + 1e-8)
            new_weights.append(w)
        return new_weights
