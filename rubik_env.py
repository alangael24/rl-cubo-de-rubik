"""
Entorno PufferEnv para el Cubo de Rubik 3x3

Este entorno sigue la API de PufferLib para entrenamiento de RL
de alto rendimiento.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pufferlib
    PUFFER_AVAILABLE = True
except ImportError:
    PUFFER_AVAILABLE = False

from rubik_cube import RubiksCube


class RubiksCubeEnv:
    """
    Entorno de Cubo de Rubik compatible con PufferLib.

    Observacion: One-hot encoding del estado del cubo (54 stickers x 6 colores = 324)
    Acciones: 12 movimientos posibles (F, F', B, B', U, U', D, D', L, L', R, R')
    """

    def __init__(
        self,
        num_envs: int = 1,
        scramble_moves: int = 20,
        max_steps: int = 200,
        reward_scale: float = 1.0,
        step_penalty: float = 0.01,
        buf=None,
    ):
        """
        Args:
            num_envs: Numero de entornos paralelos
            scramble_moves: Numero de movimientos para mezclar el cubo
            max_steps: Maximo de pasos por episodio
            reward_scale: Escala de la recompensa
            step_penalty: Penalizacion por cada paso
            buf: Buffer de PufferLib para almacenamiento in-place
        """
        self.num_envs = num_envs
        self.scramble_moves = scramble_moves
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.step_penalty = step_penalty

        # Espacios de observacion y accion
        # Observacion: estado del cubo como one-hot (54 stickers * 6 colores)
        self.single_observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(324,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(12)

        # Para compatibilidad con vectorizacion
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        # Inicializar cubos
        self.cubes = [RubiksCube() for _ in range(num_envs)]
        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_correct = np.zeros(num_envs, dtype=np.int32)

        # RNG para reproducibilidad
        self.rng = np.random.default_rng()

        # Buffers - si se proporciona buf, usar memoria compartida
        if buf is not None:
            self.observations = buf.observations
            self.actions = buf.actions
            self.rewards = buf.rewards
            self.terminals = buf.terminals
            self.truncations = buf.truncations
        else:
            self.observations = np.zeros(
                (num_envs,) + self.single_observation_space.shape,
                dtype=np.float32
            )
            self.actions = np.zeros(num_envs, dtype=np.int64)
            self.rewards = np.zeros(num_envs, dtype=np.float32)
            self.terminals = np.zeros(num_envs, dtype=bool)
            self.truncations = np.zeros(num_envs, dtype=bool)

        # Informacion adicional
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

    def _get_obs(self, env_idx: int) -> np.ndarray:
        """Obtiene la observacion one-hot para un entorno."""
        return self.cubes[env_idx].get_one_hot_state().flatten()

    def _reset_env(self, env_idx: int):
        """Resetea un entorno individual."""
        self.cubes[env_idx].reset()
        self.cubes[env_idx].scramble(self.scramble_moves, self.rng)
        self.steps[env_idx] = 0
        self.prev_correct[env_idx] = self.cubes[env_idx].count_correct_stickers()
        self.observations[env_idx] = self._get_obs(env_idx)
        self.episode_returns[env_idx] = 0.0
        self.episode_lengths[env_idx] = 0

    def reset(self, seed=None):
        """Resetea todos los entornos."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for i in range(self.num_envs):
            self._reset_env(i)

        # Limpiar terminales y truncaciones
        self.terminals[:] = False
        self.truncations[:] = False
        self.rewards[:] = 0.0

        return self.observations, {}

    def step(self, actions):
        """
        Ejecuta un paso en todos los entornos.

        Args:
            actions: Array de acciones para cada entorno

        Returns:
            observations, rewards, terminals, truncations, infos
        """
        # Limpiar recompensas y terminales al inicio del paso
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False

        infos = {}

        for i in range(self.num_envs):
            action = actions[i] if hasattr(actions, '__iter__') else actions

            # Aplicar movimiento
            self.cubes[i].apply_move(int(action))
            self.steps[i] += 1

            # Calcular recompensa
            current_correct = self.cubes[i].count_correct_stickers()
            is_solved = self.cubes[i].is_solved()

            # Recompensa por progreso (diferencia en stickers correctos)
            progress_reward = (current_correct - self.prev_correct[i]) / 54.0
            self.prev_correct[i] = current_correct

            # Penalizacion por paso
            step_reward = -self.step_penalty

            # Bonus por resolver
            if is_solved:
                self.rewards[i] = 100.0 * self.reward_scale
                self.terminals[i] = True
            else:
                self.rewards[i] = (progress_reward + step_reward) * self.reward_scale

            # Truncar si se excede el maximo de pasos
            if self.steps[i] >= self.max_steps and not self.terminals[i]:
                self.truncations[i] = True

            # Actualizar contadores de episodio
            self.episode_returns[i] += self.rewards[i]
            self.episode_lengths[i] += 1

            # Actualizar observacion
            self.observations[i] = self._get_obs(i)

            # Auto-reset si el episodio termino
            if self.terminals[i] or self.truncations[i]:
                # Guardar info del episodio antes de resetear
                infos[f'episode_return_{i}'] = float(self.episode_returns[i])
                infos[f'episode_length_{i}'] = int(self.episode_lengths[i])
                infos[f'solved_{i}'] = bool(self.terminals[i])
                self._reset_env(i)

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        """Cierra el entorno."""
        pass

    def render(self, env_idx: int = 0):
        """Renderiza el estado del cubo."""
        print(f"\n=== Entorno {env_idx} ===")
        print(f"Paso: {self.steps[env_idx]}")
        print(f"Stickers correctos: {self.cubes[env_idx].count_correct_stickers()}/54")
        print(self.cubes[env_idx])


# Wrapper para compatibilidad con PufferLib
if PUFFER_AVAILABLE:
    class PufferRubiksCubeEnv(pufferlib.PufferEnv):
        """
        Entorno PufferLib para el Cubo de Rubik.

        Este wrapper sigue la API de PufferLib para integracion
        con el sistema de entrenamiento.
        """

        def __init__(
            self,
            scramble_moves: int = 20,
            max_steps: int = 200,
            reward_scale: float = 1.0,
            step_penalty: float = 0.01,
            buf=None,
            **kwargs
        ):
            # Espacios
            self.single_observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(324,), dtype=np.float32
            )
            self.single_action_space = spaces.Discrete(12)

            super().__init__(buf=buf, **kwargs)

            self.scramble_moves = scramble_moves
            self.max_steps = max_steps
            self.reward_scale = reward_scale
            self.step_penalty = step_penalty

            # Inicializar cubo y estado
            self.cube = RubiksCube()
            self.step_count = 0
            self.prev_correct = 0
            self.rng = np.random.default_rng()

        def reset(self, seed=None):
            """Resetea el entorno."""
            if seed is not None:
                self.rng = np.random.default_rng(seed)

            self.cube.reset()
            self.cube.scramble(self.scramble_moves, self.rng)
            self.step_count = 0
            self.prev_correct = self.cube.count_correct_stickers()

            # Escribir observacion al buffer
            obs = self.cube.get_one_hot_state().flatten()
            self.observations[:] = obs

            self.terminals[:] = False
            self.truncations[:] = False
            self.rewards[:] = 0.0

            return self.observations, {}

        def step(self, actions):
            """Ejecuta un paso."""
            # Limpiar al inicio
            self.rewards[:] = 0.0
            self.terminals[:] = False
            self.truncations[:] = False

            action = actions[0] if hasattr(actions, '__iter__') else actions

            # Aplicar movimiento
            self.cube.apply_move(int(action))
            self.step_count += 1

            # Calcular recompensa
            current_correct = self.cube.count_correct_stickers()
            is_solved = self.cube.is_solved()

            progress_reward = (current_correct - self.prev_correct) / 54.0
            self.prev_correct = current_correct

            if is_solved:
                self.rewards[0] = 100.0 * self.reward_scale
                self.terminals[0] = True
            else:
                self.rewards[0] = (progress_reward - self.step_penalty) * self.reward_scale

            if self.step_count >= self.max_steps and not self.terminals[0]:
                self.truncations[0] = True

            # Actualizar observacion
            obs = self.cube.get_one_hot_state().flatten()
            self.observations[:] = obs

            infos = {}
            if self.terminals[0] or self.truncations[0]:
                infos['solved'] = bool(self.terminals[0])

            return self.observations, self.rewards, self.terminals, self.truncations, infos

        def render(self):
            """Renderiza el cubo."""
            print(f"Paso: {self.step_count}")
            print(f"Stickers correctos: {self.cube.count_correct_stickers()}/54")
            print(self.cube)

        def close(self):
            pass


def make_env(
    num_envs: int = 1,
    scramble_moves: int = 20,
    max_steps: int = 200,
    **kwargs
):
    """
    Funcion factory para crear el entorno.

    Compatible con pufferlib.vector.make()
    """
    def _make():
        return RubiksCubeEnv(
            num_envs=1,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            **kwargs
        )
    return _make


if __name__ == "__main__":
    # Test del entorno
    print("Probando RubiksCubeEnv...")

    env = RubiksCubeEnv(num_envs=2, scramble_moves=5, max_steps=50)
    obs, info = env.reset(seed=42)

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.single_action_space}")

    # Ejecutar algunos pasos
    for step in range(10):
        actions = np.random.randint(0, 12, size=2)
        obs, rewards, terminals, truncations, infos = env.step(actions)
        print(f"Step {step}: rewards={rewards}, terminals={terminals}")

    env.render(0)
    env.close()

    print("\nTest completado!")
