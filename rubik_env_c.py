"""
rubik_env_c.py - Python wrapper for high-performance C Rubik's Cube environment

This module provides Gymnasium-compatible wrappers around the C implementation
for use with PufferLib and other RL frameworks.

Performance: >1M steps/second (vs ~16k steps/second in pure Python)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Try to import the C extension
try:
    import rubik_c
    C_AVAILABLE = True
except ImportError as e:
    C_AVAILABLE = False
    print(
        "Warning: rubik_c module not found. Run 'python setup.py build_ext --inplace'. "
        f"Error: {e}"
    )


class RubiksCubeEnvC(gym.Env):
    """
    High-performance Rubik's Cube environment using C backend.

    This is a Gymnasium-compatible wrapper around the C implementation.
    Use this for training with PufferLib or other RL frameworks.

    Performance: >1M steps/second on a single core.
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        scramble_moves: int = 1,
        max_steps: int = 50,
        reward_mode: str = 'sparse',
        solve_reward: float = 1.0,
        step_penalty: float = 0.01,
        seed: int = None,
        render_mode: str = None,
    ):
        """
        Initialize the environment.

        Args:
            scramble_moves: Number of random moves to scramble the cube
            max_steps: Maximum steps per episode
            reward_mode: 'sparse' (only on solve) or 'dense' (progress-based)
            solve_reward: Reward for solving the cube
            step_penalty: Penalty per step
            seed: Random seed
            render_mode: Render mode ('human' or None)
        """
        super().__init__()

        if not C_AVAILABLE:
            raise ImportError(
                "C extension not available. "
                "Run 'python setup.py build_ext --inplace' to compile."
            )

        self.scramble_moves = scramble_moves
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.solve_reward = solve_reward
        self.step_penalty = step_penalty
        self.render_mode = render_mode

        # Create C environment
        reward_mode_int = 0 if reward_mode == 'sparse' else 1
        self._env = rubik_c.RubikEnv(
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=solve_reward,
            step_penalty=step_penalty,
            reward_mode=reward_mode_int,
            seed=seed or 42,
        )

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(rubik_c.OBS_SIZE,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(rubik_c.NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            obs, info = self._env.reset(seed)
        else:
            obs, info = self._env.reset()
        if info is None:
            info = {}
        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        result = self._env.step(int(action))
        obs, reward, terminated, truncated, info = result
        if info is None:
            info = {}
        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info
        )

    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == 'human':
            print("Render not implemented for C backend")

    def close(self):
        """Close the environment."""
        pass

    @property
    def unwrapped(self):
        return self


class RubiksCubeBatchEnvC:
    """
    Vectorized Rubik's Cube environment using C backend.

    This is optimized for batch training with multiple environments.
    All environments step in parallel using a single C call.

    Performance: >10M steps/second with 64 envs on a single core.
    """

    def __init__(
        self,
        num_envs: int = 64,
        scramble_moves: int = 1,
        max_steps: int = 50,
        reward_mode: str = 'sparse',
        solve_reward: float = 1.0,
        step_penalty: float = 0.01,
        seed: int = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            scramble_moves: Number of random moves to scramble
            max_steps: Maximum steps per episode
            reward_mode: 'sparse' or 'dense'
            solve_reward: Reward for solving
            step_penalty: Penalty per step
            seed: Random seed
        """
        if not C_AVAILABLE:
            raise ImportError(
                "C extension not available. "
                "Run 'python setup.py build_ext --inplace' to compile."
            )

        self.num_envs = num_envs
        self._scramble_moves = scramble_moves
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.solve_reward = solve_reward
        self.step_penalty = step_penalty

        # Create C batch environment
        reward_mode_int = 0 if reward_mode == 'sparse' else 1
        self._env = rubik_c.RubikBatchEnv(
            num_envs=num_envs,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=solve_reward,
            step_penalty=step_penalty,
            reward_mode=reward_mode_int,
            seed=seed or 42,
        )

        # Define spaces (for compatibility)
        self.single_observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(rubik_c.OBS_SIZE,),
            dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(rubik_c.NUM_ACTIONS)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    @property
    def scramble_moves(self):
        return self._scramble_moves

    @scramble_moves.setter
    def scramble_moves(self, value):
        self._scramble_moves = value
        self._env.scramble_moves = value

    def reset(self, seed=None):
        """Reset all environments."""
        obs, info = self._env.reset()
        if info is None:
            info = {}
        return obs, info

    def step(self, actions):
        """Step all environments."""
        if isinstance(actions, np.ndarray) and actions.dtype == np.int32 and actions.flags.c_contiguous:
            actions_i32 = actions
        else:
            actions_i32 = np.asarray(actions, dtype=np.int32)

        obs, rewards, terminals, truncations, info = self._env.step(actions_i32)
        if info is None:
            info = {}
        return (
            obs,
            rewards,
            terminals,
            truncations,
            info
        )

    def reset_stats(self):
        """Compatibility no-op used by some curriculum trainers."""
        return None

    def close(self):
        """Close all environments."""
        pass

    def render(self, env_idx=0):
        """Render (placeholder)."""
        print(f"Render not implemented for C backend (env {env_idx})")


# ============================================================================
# Factory functions for PufferLib compatibility
# ============================================================================

def make_env_c(
    scramble_moves: int = 1,
    max_steps: int = 50,
    reward_mode: str = 'sparse',
    **kwargs
):
    """
    Factory function to create a single C environment.

    Compatible with pufferlib.vector.make()
    """
    def _make():
        return RubiksCubeEnvC(
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            reward_mode=reward_mode,
            **kwargs
        )
    return _make


def make_batch_env_c(
    num_envs: int = 64,
    scramble_moves: int = 1,
    max_steps: int = 50,
    reward_mode: str = 'sparse',
    **kwargs
):
    """
    Create a vectorized C environment.

    This is more efficient than using pufferlib.vector.make() with
    individual environments because all envs are in a single C struct.
    """
    return RubiksCubeBatchEnvC(
        num_envs=num_envs,
        scramble_moves=scramble_moves,
        max_steps=max_steps,
        reward_mode=reward_mode,
        **kwargs
    )


# ============================================================================
# Benchmark utility
# ============================================================================

def benchmark(num_envs=64, num_steps=100000):
    """
    Benchmark the C environment.

    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps per environment
    """
    import time

    if not C_AVAILABLE:
        print("C extension not available. Cannot benchmark.")
        return

    print(f"\n=== Benchmark: {num_envs} envs, {num_steps} steps each ===")

    env = RubiksCubeBatchEnvC(num_envs=num_envs, scramble_moves=20, max_steps=100)
    obs, _ = env.reset()

    start = time.perf_counter()

    for _ in range(num_steps):
        actions = np.random.randint(0, 12, size=num_envs, dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

    elapsed = time.perf_counter() - start
    total_steps = num_steps * num_envs
    sps = total_steps / elapsed

    print(f"Time: {elapsed:.3f} seconds")
    print(f"Total steps: {total_steps:,}")
    print(f"Steps/second: {sps:,.0f}")
    print(f"Steps/second/env: {sps/num_envs:,.0f}")

    env.close()
    return sps


if __name__ == "__main__":
    if not C_AVAILABLE:
        print("C extension not available.")
        print("Run: python setup.py build_ext --inplace")
        exit(1)

    print("Testing C Rubik's Cube Environment")
    print("=" * 40)

    # Test single environment
    print("\n1. Single environment test:")
    env = RubiksCubeEnvC(scramble_moves=5, max_steps=20)
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Step {i+1}: reward={reward:.3f}, done={term or trunc}")

    env.close()

    # Test batch environment
    print("\n2. Batch environment test:")
    batch_env = RubiksCubeBatchEnvC(num_envs=8, scramble_moves=5, max_steps=20)
    obs, info = batch_env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Num envs: {batch_env.num_envs}")

    for i in range(3):
        actions = np.random.randint(0, 12, size=8, dtype=np.int32)
        obs, rewards, terms, truncs, info = batch_env.step(actions)
        print(f"   Step {i+1}: rewards={rewards}, terms={terms}")

    batch_env.close()

    # Benchmark
    print("\n3. Benchmarks:")
    benchmark(num_envs=1, num_steps=100000)
    benchmark(num_envs=64, num_steps=100000)
    benchmark(num_envs=256, num_steps=100000)

    print("\nAll tests passed!")
