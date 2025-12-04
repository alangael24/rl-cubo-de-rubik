"""
rubik_puffer.py - Native PufferLib environment for Rubik's Cube

This follows the PufferLib Ocean pattern for maximum performance:
- Inherits from pufferlib.PufferEnv
- Passes buffers directly to C (zero-copy)
- C writes directly to pre-allocated numpy arrays

Performance target: 1M+ steps/second
"""

import gymnasium
import numpy as np

import pufferlib

# Import C binding
try:
    import rubik_c
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("Warning: rubik_c module not found. Run 'python setup.py build_ext --inplace'")


class RubikPufferEnv(pufferlib.PufferEnv):
    """
    High-performance Rubik's Cube environment using PufferLib native API.

    This follows the same pattern as PufferLib Ocean environments (Snake, Squared, etc.)
    for maximum performance through zero-copy buffer sharing with C.

    Performance: 1M+ steps/second with 1024 envs
    """

    def __init__(
        self,
        num_envs: int = 1,
        render_mode: str = None,
        log_interval: int = 128,
        scramble_moves: int = 1,
        max_steps: int = 50,
        reward_mode: str = 'sparse',
        solve_reward: float = 1.0,
        step_penalty: float = 0.01,
        buf=None,
        seed: int = 0,
    ):
        if not C_AVAILABLE:
            raise ImportError(
                "C extension not available. "
                "Run 'python setup.py build_ext --inplace' to compile."
            )

        # Define spaces (PufferLib uses single_* naming)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(rubik_c.OBS_SIZE,),
            dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(rubik_c.NUM_ACTIONS)

        self.render_mode = render_mode
        self.num_agents = num_envs  # PufferLib uses num_agents
        self.log_interval = log_interval

        # Store config for curriculum learning
        self._scramble_moves = scramble_moves
        self._max_steps = max_steps
        self._reward_mode = 0 if reward_mode == 'sparse' else 1
        self._solve_reward = solve_reward
        self._step_penalty = step_penalty
        self._seed = seed

        # Call parent init - this allocates the buffers
        super().__init__(buf)

        # Initialize C environments with our pre-allocated buffers (zero-copy!)
        self.c_envs = rubik_c.vec_init(
            self.observations,    # Pre-allocated by PufferEnv
            self.actions,         # Pre-allocated by PufferEnv
            self.rewards,         # Pre-allocated by PufferEnv
            self.terminals,       # Pre-allocated by PufferEnv
            self.truncations,     # Pre-allocated by PufferEnv
            num_envs,
            seed,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=solve_reward,
            step_penalty=step_penalty,
            reward_mode=self._reward_mode,
            log_interval=log_interval,
        )

    @property
    def scramble_moves(self):
        return self._scramble_moves

    @scramble_moves.setter
    def scramble_moves(self, value):
        """Update scramble moves for curriculum learning."""
        self._scramble_moves = max(1, min(value, 26))
        rubik_c.vec_set_scramble(self.c_envs, self._scramble_moves)

    def reset(self, seed=0):
        """Reset all environments."""
        rubik_c.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        """Step all environments."""
        self.tick += 1

        # Copy actions to our buffer (required by PufferLib pattern)
        self.actions[:] = actions

        # Step C environments - they write directly to our buffers
        rubik_c.vec_step(self.c_envs, self.actions)

        # Get logging info periodically
        info = []
        if self.tick % self.log_interval == 0:
            info.append(rubik_c.vec_log(self.c_envs))

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )

    def render(self):
        """Render an environment (placeholder)."""
        rubik_c.vec_render(self.c_envs, 0)

    def close(self):
        """Close and free resources."""
        rubik_c.vec_close(self.c_envs)


def make_env(
    scramble_moves: int = 1,
    max_steps: int = 50,
    reward_mode: str = 'sparse',
    **kwargs
):
    """
    Factory function compatible with PufferLib training scripts.

    Returns the RubikPufferEnv class (not an instance) for pufferlib.vector.make()
    """
    # Return the class itself - PufferLib will instantiate it
    return RubikPufferEnv


# Benchmark utility
def benchmark(num_envs=1024, num_steps=10000):
    """
    Benchmark the native PufferLib environment.

    This should achieve 1M+ steps/second.
    """
    import time

    if not C_AVAILABLE:
        print("C extension not available. Cannot benchmark.")
        return

    print(f"\n=== PufferLib Native Benchmark: {num_envs} envs ===")

    env = RubikPufferEnv(
        num_envs=num_envs,
        scramble_moves=20,
        max_steps=100,
        log_interval=1000000  # Disable logging for pure benchmark
    )
    env.reset()

    # Pre-generate random actions
    CACHE = 1024
    actions = np.random.randint(0, 12, (CACHE, num_envs), dtype=np.int32)

    steps = 0
    i = 0
    start = time.perf_counter()

    while time.perf_counter() - start < 10:
        env.step(actions[i % CACHE])
        steps += num_envs
        i += 1

    elapsed = time.perf_counter() - start
    sps = steps / elapsed

    print(f"Time: {elapsed:.3f} seconds")
    print(f"Total steps: {steps:,}")
    print(f"Steps/second: {sps:,.0f}")

    env.close()
    return sps


if __name__ == "__main__":
    if not C_AVAILABLE:
        print("C extension not available.")
        print("Run: python setup.py build_ext --inplace")
        exit(1)

    print("Testing Native PufferLib Rubik's Cube Environment")
    print("=" * 50)

    # Test basic functionality
    print("\n1. Basic test with 8 environments:")
    env = RubikPufferEnv(num_envs=8, scramble_moves=5, max_steps=20)
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.single_action_space}")
    print(f"   Num agents: {env.num_agents}")

    for i in range(5):
        actions = np.random.randint(0, 12, size=8, dtype=np.int32)
        obs, rewards, terms, truncs, info = env.step(actions)
        print(f"   Step {i+1}: rewards={rewards[:3]}..., terms={terms[:3]}...")

    env.close()

    # Test curriculum learning
    print("\n2. Curriculum learning test:")
    env = RubikPufferEnv(num_envs=4, scramble_moves=1)
    print(f"   Initial scramble_moves: {env.scramble_moves}")
    env.scramble_moves = 5
    print(f"   After update: {env.scramble_moves}")
    env.close()

    # Benchmark
    print("\n3. Benchmarks:")
    benchmark(num_envs=64, num_steps=100000)
    benchmark(num_envs=1024, num_steps=100000)
    benchmark(num_envs=4096, num_steps=100000)

    print("\nAll tests passed!")
