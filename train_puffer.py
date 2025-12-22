"""
Entrenamiento del Cubo de Rubik con PufferLib CLI oficial.

Uso:
    # Entrenamiento basico
    python train_puffer.py

    # Con GPU
    python train_puffer.py --device cuda

    # Cambiar hiperparametros
    python train_puffer.py --train.total_timesteps 50_000_000

    # Ver todas las opciones
    python train_puffer.py --help
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.emulation
from pufferlib import pufferl

from rubik_env import GymnasiumRubiksCubeEnv


# ============================================================================
# ResNet Policy para PufferLib
# ============================================================================

class ResidualBlock(nn.Module):
    """Bloque residual con skip connection."""
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
        x = F.relu(x + residual)
        return x


class Policy(nn.Module):
    """
    ResNet Policy para el Cubo de Rubik - Compatible con PufferLib.

    Arquitectura basada en DeepCubeA:
    - Input: One-hot encoding (324 = 54 stickers * 6 colores)
    - Bloques residuales
    - Heads separados para actor (policy) y critic (value)
    """
    def __init__(self, env, hidden_size=512, num_residual_blocks=4):
        super().__init__()

        # Obtener dimensiones del entorno
        obs_size = np.prod(env.single_observation_space.shape)
        action_size = env.single_action_space.n

        self.hidden_size = hidden_size

        # Input projection
        self.input_fc = pufferlib.pytorch.layer_init(
            nn.Linear(obs_size, hidden_size)
        )
        self.input_ln = nn.LayerNorm(hidden_size)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_residual_blocks)
        ])

        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_out = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size // 2, action_size), std=0.01
        )

        # Critic head (value)
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_out = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size // 2, 1), std=1.0
        )

    def forward(self, observations, state=None):
        """Forward pass - requerido por PufferLib."""
        return self.forward_eval(observations, state)

    def forward_eval(self, observations, state=None):
        """Forward pass para evaluacion."""
        hidden = self.encode_observations(observations)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def encode_observations(self, observations, state=None):
        """Codifica observaciones en estados ocultos."""
        batch_size = observations.shape[0]
        x = observations.view(batch_size, -1).float()

        # Input projection
        x = F.relu(self.input_ln(self.input_fc(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        return x

    def decode_actions(self, hidden):
        """Decodifica estados ocultos en acciones y valores."""
        # Actor (policy logits)
        actor = F.relu(self.actor_fc(hidden))
        logits = self.actor_out(actor)

        # Critic (value)
        critic = F.relu(self.critic_fc(hidden))
        value = self.critic_out(critic)

        return logits, value


class Recurrent(pufferlib.models.LSTMWrapper):
    """Wrapper LSTM para la policy (opcional, para problemas que requieren memoria)."""
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size=input_size, hidden_size=hidden_size)


# ============================================================================
# Funciones para crear el entorno
# ============================================================================

def make_env(
    scramble_moves: int = 1,
    max_steps: int = 50,
    reward_mode: str = 'sparse',
    buf=None,
    **kwargs
):
    """
    Factory function para crear el entorno del Cubo de Rubik.
    Compatible con pufferlib.vector.make()
    """
    env = GymnasiumRubiksCubeEnv(
        scramble_moves=scramble_moves,
        max_steps=max_steps,
        reward_mode=reward_mode,
    )
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def env_creator(env_name='rubiks_cube'):
    """Retorna la funcion factory del entorno."""
    return make_env


# ============================================================================
# Entrenamiento con PufferLib
# ============================================================================

def train_simple():
    """Entrenamiento simple usando PuffeRL directamente."""
    print("=" * 60)
    print("ENTRENAMIENTO CUBO DE RUBIK CON PUFFERLIB")
    print("=" * 60)

    # Crear vectorized environment
    vecenv = pufferlib.vector.make(
        env_creator=make_env,
        env_kwargs={
            'scramble_moves': 1,
            'max_steps': 50,
            'reward_mode': 'sparse',
        },
        num_envs=4,           # Procesos paralelos
        num_workers=4,
        batch_size=1,
        backend=pufferlib.vector.Multiprocessing,
    )

    # Crear policy
    policy = Policy(vecenv.driver_env, hidden_size=512, num_residual_blocks=4)

    # Mover a GPU si disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = policy.to(device)

    print(f"\nPolicy ResNet:")
    print(f"  Hidden size: 512")
    print(f"  Residual blocks: 4")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")

    # Cargar config default y modificar
    args = pufferl.load_config('default')
    args['train']['env'] = 'rubiks_cube'
    args['train']['total_timesteps'] = 10_000_000
    args['train']['learning_rate'] = 3e-4
    args['train']['device'] = device

    # Crear trainer
    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    print(f"\nIniciando entrenamiento...")
    print(f"  Total timesteps: {args['train']['total_timesteps']:,}")
    print("=" * 60)

    # Loop de entrenamiento
    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            logs = trainer.train()

            # Mostrar progreso cada 10 epochs
            if trainer.epoch % 10 == 0:
                print(f"Epoch {trainer.epoch}/{trainer.total_epochs} | "
                      f"Steps: {trainer.global_step:,}")

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")

    # Mostrar estadisticas finales
    trainer.print_dashboard()

    # Guardar modelo
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': trainer.epoch,
        'global_step': trainer.global_step,
    }, 'rubik_puffer.pt')
    print(f"\nModelo guardado en rubik_puffer.pt")

    trainer.close()


def train_with_curriculum():
    """Entrenamiento con curriculum learning (aumentar dificultad progresivamente)."""
    print("=" * 60)
    print("ENTRENAMIENTO CON CURRICULUM LEARNING")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    current_scramble = 1
    max_scramble = 20
    success_threshold = 0.8
    timesteps_per_level = 1_000_000

    # Crear policy
    dummy_env = make_env(scramble_moves=1)
    policy = Policy(dummy_env, hidden_size=512, num_residual_blocks=4).to(device)
    dummy_env.close()

    print(f"Device: {device}")
    print(f"Scramble inicial: {current_scramble}")
    print(f"Scramble maximo: {max_scramble}")
    print("=" * 60)

    while current_scramble <= max_scramble:
        print(f"\n>>> NIVEL {current_scramble}: {current_scramble} scramble moves <<<")

        # Crear environment con dificultad actual
        vecenv = pufferlib.vector.make(
            env_creator=make_env,
            env_kwargs={
                'scramble_moves': current_scramble,
                'max_steps': max(50, current_scramble * 3),
                'reward_mode': 'sparse',
            },
            num_envs=4,
            num_workers=4,
            batch_size=1,
            backend=pufferlib.vector.Multiprocessing,
        )

        # Configurar trainer
        args = pufferl.load_config('default')
        args['train']['env'] = f'rubiks_cube_scramble_{current_scramble}'
        args['train']['total_timesteps'] = timesteps_per_level
        args['train']['learning_rate'] = 3e-4
        args['train']['device'] = device

        trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

        # Entrenar este nivel
        try:
            while trainer.epoch < trainer.total_epochs:
                trainer.evaluate()
                logs = trainer.train()
        except KeyboardInterrupt:
            print("\nInterrumpido")
            break

        trainer.close()

        # Avanzar al siguiente nivel
        current_scramble += 1
        print(f"Avanzando a scramble {current_scramble}...")

    # Guardar modelo final
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'scramble_level': current_scramble - 1,
    }, 'rubik_puffer_curriculum.pt')
    print(f"\nModelo guardado en rubik_puffer_curriculum.pt")


# ============================================================================
# Main con argumentos CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar Cubo de Rubik con PufferLib")
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["simple", "curriculum"],
                        help="Modo de entrenamiento")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device para entrenamiento")

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "simple":
        train_simple()
    else:
        train_with_curriculum()
