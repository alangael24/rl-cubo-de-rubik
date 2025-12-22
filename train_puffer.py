"""
Entrenamiento del Cubo de Rubik con PufferLib + Backend C de alto rendimiento.
Optimizado para >1M steps/segundo.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium

import pufferlib
import pufferlib.pytorch
from pufferlib import pufferl

# Intentar importar backend C
try:
    import rubik_c
    C_BACKEND = True
    print(">>> USANDO BACKEND C (OPTIMIZADO) <<<")
except ImportError:
    C_BACKEND = False
    print(">>> BACKEND C NO DISPONIBLE <<<")
    print(">>> Ejecuta: python setup.py build_ext --inplace <<<")


# ============================================================================
# PufferEnv optimizado (siguiendo patron de Snake)
# ============================================================================

class RubiksPufferEnv(pufferlib.PufferEnv):
    """
    PufferEnv nativo optimizado para maximo rendimiento.
    Evita copias de datos y loops en Python.
    """

    def __init__(self, num_envs=4096, scramble_moves=1, max_steps=50,
                 reward_mode='sparse', report_interval=64, buf=None, seed=0, **kwargs):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(324,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(12)
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0

        # Inicializar buffers via parent class
        super().__init__(buf)

        # Crear entorno C con punteros a nuestros buffers
        reward_mode_int = 0 if reward_mode == 'sparse' else 1
        self._c_env = rubik_c.RubikBatchEnv(
            num_envs=num_envs,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=1.0,
            step_penalty=0.01,
            reward_mode=reward_mode_int,
            seed=seed,
        )

        self._num_envs = num_envs
        self._scramble_moves = scramble_moves

        # Stats para logging (solo cada report_interval)
        self._total_solved = 0
        self._total_episodes = 0

    @property
    def emulated(self):
        return None

    def reset(self, seed=None):
        self.tick = 0
        obs, _ = self._c_env.reset()
        # Escribir directamente al buffer compartido
        np.copyto(self.observations, obs)
        self.rewards.fill(0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        self.masks.fill(True)
        self._total_solved = 0
        self._total_episodes = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1

        # Llamar a C - esto es lo mas rapido posible
        obs, rewards, terms, truncs, _ = self._c_env.step(actions.astype(np.int32))

        # Copiar resultados a buffers compartidos (inevitable sin modificar el binding C)
        np.copyto(self.observations, obs)
        np.copyto(self.rewards, rewards)
        np.copyto(self.terminals, terms)
        np.copyto(self.truncations, truncs)

        # Info solo cada report_interval (como Snake)
        info = []
        if self.tick % self.report_interval == 0:
            dones = terms | truncs
            n_done = np.sum(dones)
            if n_done > 0:
                n_solved = np.sum(terms)
                self._total_episodes += n_done
                self._total_solved += n_solved
                info.append({
                    'episode_return': float(np.mean(rewards[dones])) if n_done > 0 else 0,
                    'episode_length': float(self.tick),
                    'solved': float(self._total_solved) / max(1, self._total_episodes),
                })

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        pass


# ============================================================================
# ResNet Policy (compacta)
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)

    def forward(self, x):
        return F.relu(self.ln2(self.fc2(F.relu(self.ln1(self.fc1(x))))) + x)


class Policy(nn.Module):
    def __init__(self, env, hidden_size=512, num_blocks=4):
        super().__init__()
        obs = np.prod(env.single_observation_space.shape)
        act = env.single_action_space.n

        self.input_fc = pufferlib.pytorch.layer_init(nn.Linear(obs, hidden_size))
        self.input_ln = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, act), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x, state=None):
        x = F.relu(self.input_ln(self.input_fc(x.float().view(x.shape[0], -1))))
        for block in self.blocks:
            x = block(x)
        return self.actor(x), self.critic(x)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not C_BACKEND:
        print("ERROR: Backend C no disponible")
        exit(1)

    print("=" * 60)
    print("ENTRENAMIENTO CUBO DE RUBIK - OPTIMIZADO")
    print("=" * 60)

    args = pufferl.load_config('default')
    args['train']['env'] = 'rubiks_cube'
    args['train']['total_timesteps'] = args['train'].get('total_timesteps', 10_000_000)
    args['train']['learning_rate'] = 3e-4
    args['train']['minibatch_size'] = 4096
    args['train']['bptt_horizon'] = 16

    NUM_ENVS = 8192  # Mas envs = mas throughput

    vecenv = pufferlib.vector.make(
        RubiksPufferEnv,
        env_kwargs={'num_envs': NUM_ENVS, 'scramble_moves': 1, 'max_steps': 50},
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )

    device = args['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(vecenv, hidden_size=512, num_blocks=4).to(device)

    print(f"\n  Params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Envs: {NUM_ENVS}")
    print("=" * 60)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            trainer.train()
    except KeyboardInterrupt:
        print("\nInterrumpido")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'rubik_puffer.pt')
    trainer.close()
