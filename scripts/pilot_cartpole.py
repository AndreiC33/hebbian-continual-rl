"""Pilot experiment: evolve Hebbian rules with OpenAI-ES on CartPole (gymnax).

Every episode starts from NEW random weights; the agent can only succeed if the
evolved rules make the weights self-organize into a working controller.

Run from the repository root:
    python -m scripts.pilot_cartpole
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import gymnax
import jax
import jax.numpy as jnp
import optax
from evosax.algorithms import Open_ES

from hcrl import HebbianNet


def make_rollout(env, env_params, net: HebbianNet, max_steps: int, plastic: bool = True):
    """Return a jitted function: (rule_vector, key) -> episode return."""

    def rollout(flat_rules, key):
        rules = net.unflatten_rules(flat_rules)
        k_w, k_reset, k_steps = jax.random.split(key, 3)
        weights = net.init_weights(k_w)
        obs, env_state = env.reset(k_reset, env_params)

        def step(carry, k):
            obs, env_state, weights, total, alive = carry
            out, acts = net.forward(weights, obs)
            action = jnp.argmax(out)
            out_step = env.step(k, env_state, action, env_params)
            obs, env_state, reward = out_step[:3]
            if len(out_step) == 6:  # gymnax >= 1.0: (obs, state, reward, terminated, truncated, info)
                done = jnp.logical_or(out_step[3], out_step[4])
            else:  # gymnax < 1.0: (obs, state, reward, done, info)
                done = out_step[3]
            done = done.astype(jnp.float32)
            if plastic:
                weights = net.hebbian_update(weights, rules, acts)
            total = total + reward * alive
            alive = alive * (1.0 - done)
            return (obs, env_state, weights, total, alive), None

        keys = jax.random.split(k_steps, max_steps)
        (_, _, _, total, _), _ = jax.lax.scan(step, (obs, env_state, weights, 0.0, 1.0), keys)
        return total

    return rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--inits", type=int, default=3, help="random weight inits per fitness evaluation")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results")
    args = p.parse_args()

    env, env_params = gymnax.make("CartPole-v1")
    max_steps = 500
    net = HebbianNet((4, 32, 32, 2))
    print(f"Synapses: {net.num_synapses}, evolved rule parameters: {net.num_rule_params}")

    rollout = make_rollout(env, env_params, net, max_steps, plastic=True)
    static_rollout = make_rollout(env, env_params, net, max_steps, plastic=False)

    # fitness of one genome = mean return over several fresh random weight initialisations
    def fitness_one(genome, key):
        return jnp.mean(jax.vmap(rollout, in_axes=(None, 0))(genome, jax.random.split(key, args.inits)))

    eval_pop = jax.jit(jax.vmap(fitness_one, in_axes=(0, None)))

    strategy = Open_ES(
        population_size=args.pop,
        solution=jnp.zeros(net.num_rule_params),
        optimizer=optax.adam(args.lr),
        std_schedule=optax.constant_schedule(args.sigma),
    )
    params = strategy.default_params
    key = jax.random.key(args.seed)
    key, k_init = jax.random.split(key)
    state = strategy.init(k_init, jnp.zeros(net.num_rule_params), params)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    for gen in range(args.generations):
        key, k_ask, k_eval, k_tell = jax.random.split(key, 4)
        pop, state = strategy.ask(k_ask, state, params)
        returns = eval_pop(pop, k_eval)  # same env/weight seeds for all members (common random numbers)
        state, _ = strategy.tell(k_tell, pop, -returns, state, params)  # evosax minimises
        log.append((gen, float(returns.mean()), float(returns.max())))
        if gen % 10 == 0 or gen == args.generations - 1:
            print(f"gen {gen:4d} | mean return {returns.mean():6.1f} | best {returns.max():6.1f} | {time.time() - t0:5.1f}s")

    # Final test: evolved rules on 100 brand-new random weight initialisations
    mean_rules = strategy.get_mean(state)
    test_keys = jax.random.split(jax.random.key(10_000 + args.seed), 100)
    hebb_test = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))(mean_rules, test_keys)
    static_test = jax.jit(jax.vmap(static_rollout, in_axes=(None, 0)))(mean_rules, test_keys)
    print(f"\nTest on 100 new random inits | Hebbian (plastic): {hebb_test.mean():.1f} ± {hebb_test.std():.1f}"
          f" | same random weights, plasticity OFF: {static_test.mean():.1f} ± {static_test.std():.1f}")

    with open(out_dir / "pilot_cartpole_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_return", "best_return"])
        w.writerows(log)
    with open(out_dir / "pilot_cartpole_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_return", "std_return", "n"])
        w.writerow(["hebbian_plastic", float(hebb_test.mean()), float(hebb_test.std()), 100])
        w.writerow(["random_weights_no_plasticity", float(static_test.mean()), float(static_test.std()), 100])
    jnp.save(out_dir / "pilot_cartpole_rules.npy", mean_rules)
    print(f"Saved logs to {out_dir}/")


if __name__ == "__main__":
    main()
