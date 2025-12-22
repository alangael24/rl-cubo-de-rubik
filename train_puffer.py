"""
Entrenamiento del Cubo de Rubik con PufferLib + Backend C de alto rendimiento.

Usa el backend C nativo para >1M steps/segundo.

Uso:
    python train_puffer.py --train.device cuda
    python train_puffer.py --train.total-timesteps 50000000
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.pytorch
from pufferlib import pufferl

# Intentar importar backend C
try:
    from rubik_env_c import RubiksCubeBatchEnvC
    import rubik_c
    C_BACKEND = True
    print(">>> USANDO BACKEND C (OPTIMIZADO) <<<")
except ImportError:
    C_BACKEND = False
    print(">>> USANDO BACKEND PYTHON (LENTO) <<<")
    print(">>> Ejecuta: python setup.py build_ext --inplace <<<")


# ============================================================================
# PufferEnv nativo para el backend C (maximo rendimiento)
# ============================================================================

class RubiksPufferEnv(pufferlib.PufferEnv):
    """
    Wrapper PufferEnv nativo para el backend C.
    Esto evita el overhead de la vectorizacion de PufferLib.
    """

    def __init__(self, num_envs=1024, scramble_moves=1, max_steps=50,
                 reward_mode='sparse', buf=None, **kwargs):
        # Crear el entorno C batch
        reward_mode_int = 0 if reward_mode == 'sparse' else 1
        self._c_env = rubik_c.RubikBatchEnv(
            num_envs=num_envs,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=1.0,
            step_penalty=0.01,
            reward_mode=reward_mode_int,
            seed=42,
        )

        self._num_envs = num_envs
        self._scramble_moves = scramble_moves
        self.max_steps = max_steps

        # Espacios
        self.single_observation_space = pufferlib.spaces.Box(
            low=0.0, high=1.0, shape=(324,), dtype=np.float32
        )
        self.single_action_space = pufferlib.spaces.Discrete(12)

        # Para PufferLib
        self.num_agents = num_envs
        self.tick = 0

        # Buffers
        if buf is None:
            self.observations = np.zeros((num_envs, 324), dtype=np.float32)
            self.rewards = np.zeros(num_envs, dtype=np.float32)
            self.terminals = np.zeros(num_envs, dtype=bool)
            self.truncations = np.zeros(num_envs, dtype=bool)
            self.masks = np.ones(num_envs, dtype=bool)
            self.actions = np.zeros(num_envs, dtype=np.int32)
        else:
            self.observations = buf['observations']
            self.rewards = buf['rewards']
            self.terminals = buf['terminals']
            self.truncations = buf['truncations']
            self.masks = buf['masks']
            self.actions = buf['actions']

        # Stats
        self._episode_returns = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self._solved_count = 0
        self._episode_count = 0

    @property
    def emulated(self):
        return None

    @property
    def scramble_moves(self):
        return self._scramble_moves

    @scramble_moves.setter
    def scramble_moves(self, value):
        self._scramble_moves = value
        self._c_env.scramble_moves = value

    def reset(self, seed=None):
        obs, _ = self._c_env.reset()
        self.observations[:] = obs
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        self._episode_returns[:] = 0
        self._episode_lengths[:] = 0
        self.tick = 0
        return self.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        obs, rewards, terms, truncs, info = self._c_env.step(actions.astype(np.int32))

        self.observations[:] = obs
        self.rewards[:] = rewards
        self.terminals[:] = terms
        self.truncations[:] = truncs

        # Actualizar stats
        self._episode_returns += rewards
        self._episode_lengths += 1

        dones = terms | truncs
        if np.any(dones):
            for i in np.where(dones)[0]:
                self._episode_count += 1
                if terms[i]:
                    self._solved_count += 1
                self._episode_returns[i] = 0
                self._episode_lengths[i] = 0

        self.tick += 1

        # Info para logging
        info = {
            'episode_return': float(np.mean(self._episode_returns)),
            'episode_length': float(np.mean(self._episode_lengths)),
            'solved': self._solved_count / max(1, self._episode_count),
        }

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        pass


# ============================================================================
# ResNet Policy
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


class Policy(nn.Module):
    def __init__(self, env, hidden_size=512, num_residual_blocks=4):
        super().__init__()
        obs_size = np.prod(env.single_observation_space.shape)
        action_size = env.single_action_space.n
        self.hidden_size = hidden_size

        self.input_fc = pufferlib.pytorch.layer_init(nn.Linear(obs_size, hidden_size))
        self.input_ln = nn.LayerNorm(hidden_size)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_residual_blocks)
        ])
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_out = pufferlib.pytorch.layer_init(nn.Linear(hidden_size // 2, action_size), std=0.01)
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_out = pufferlib.pytorch.layer_init(nn.Linear(hidden_size // 2, 1), std=1.0)

    def forward(self, observations, state=None):
        batch_size = observations.shape[0]
        x = observations.view(batch_size, -1).float()
        x = F.relu(self.input_ln(self.input_fc(x)))
        for block in self.residual_blocks:
            x = block(x)
        logits = self.actor_out(F.relu(self.actor_fc(x)))
        value = self.critic_out(F.relu(self.critic_fc(x)))
        return logits, value

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not C_BACKEND:
        print("ERROR: Backend C no disponible. Ejecuta:")
        print("  python setup.py build_ext --inplace")
        exit(1)

    print("=" * 60)
    print("ENTRENAMIENTO CUBO DE RUBIK CON PUFFERLIB")
    print("=" * 60)

    # Cargar config
    args = pufferl.load_config('default')
    args['train']['env'] = 'rubiks_cube'
    args['train']['total_timesteps'] = args['train'].get('total_timesteps', 10_000_000)
    args['train']['learning_rate'] = args['train'].get('learning_rate', 3e-4)
    args['train']['minibatch_size'] = 4096

    # Crear entorno nativo (esto es lo que da 1M+ SPS)
    NUM_ENVS = 4096  # Muchos entornos en paralelo en C

    vecenv = pufferlib.vector.make(
        RubiksPufferEnv,
        env_kwargs={
            'num_envs': NUM_ENVS,
            'scramble_moves': 1,
            'max_steps': 50,
            'reward_mode': 'sparse',
        },
        num_envs=1,  # Solo 1 "worker" porque el env ya es vectorizado internamente
        backend=pufferlib.PufferEnv,  # Backend nativo
    )

    # Policy
    device = args['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(vecenv, hidden_size=512, num_residual_blocks=4).to(device)

    print(f"\nPolicy ResNet:")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Num envs (C batch): {NUM_ENVS}")
    print(f"  Total timesteps: {args['train']['total_timesteps']:,}")
    print("=" * 60)

    # Train
    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    print("\nIniciando entrenamiento...")
    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            trainer.train()
    except KeyboardInterrupt:
        print("\nInterrumpido")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'rubik_puffer.pt')
    print("\nModelo guardado en rubik_puffer.pt")
    trainer.close()
