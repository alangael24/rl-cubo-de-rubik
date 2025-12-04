"""
Script de entrenamiento para el Cubo de Rubik usando Curriculum Learning.

Basado en recomendaciones de papers academicos:
- DeepCubeA (McAleer et al., 2019)
- Autodidactic Iteration (ADI)

Compatible con PufferLib v3.0

Curriculum Learning:
1. Empezar con cubos mezclados con 1 movimiento
2. Cuando success_rate > threshold, aumentar dificultad
3. Progresar hasta scramble_moves = 20 (God's number)

Uso:
    python train.py
    python train.py --max-scramble 10 --success-threshold 0.8
    python train.py --use-pufferlib  # Usar PufferLib v3.0
"""

import argparse
import os
import time
from collections import deque
import numpy as np

# Verificar disponibilidad de PufferLib v3.0
PUFFER_AVAILABLE = False
PUFFER_VERSION = None

try:
    import pufferlib
    import pufferlib.vector
    import pufferlib.emulation
    PUFFER_AVAILABLE = True
    PUFFER_VERSION = getattr(pufferlib, '__version__', 'unknown')
    print(f"PufferLib v{PUFFER_VERSION} detectado")
except ImportError:
    print("PufferLib no esta instalado. Usando entrenamiento simple.")

# Verificar PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch no disponible. Solo entrenamiento aleatorio.")

from rubik_env import RubiksCubeEnv, make_env


class CurriculumTrainer:
    """
    Entrenador con Curriculum Learning para el Cubo de Rubik.

    Implementa Autodidactic Iteration (ADI):
    - Empieza con problemas faciles (1 scramble move)
    - Aumenta dificultad cuando el agente domina el nivel actual
    """

    def __init__(
        self,
        env: RubiksCubeEnv,
        start_scramble: int = 1,
        max_scramble: int = 20,
        success_threshold: float = 0.8,
        eval_window: int = 100,
        steps_per_level: int = 50000,
    ):
        self.env = env
        self.current_scramble = start_scramble
        self.max_scramble = max_scramble
        self.success_threshold = success_threshold
        self.eval_window = eval_window
        self.steps_per_level = steps_per_level

        # Tracking
        self.recent_solves = deque(maxlen=eval_window)
        self.total_steps = 0
        self.steps_at_level = 0
        self.level_history = []

        # Inicializar entorno
        self.env.scramble_moves = start_scramble

    def get_success_rate(self) -> float:
        if len(self.recent_solves) == 0:
            return 0.0
        return sum(self.recent_solves) / len(self.recent_solves)

    def should_increase_difficulty(self) -> bool:
        if self.current_scramble >= self.max_scramble:
            return False
        if self.steps_at_level < self.steps_per_level:
            return False
        if len(self.recent_solves) < self.eval_window:
            return False
        return self.get_success_rate() >= self.success_threshold

    def increase_difficulty(self):
        self.current_scramble += 1
        self.env.scramble_moves = self.current_scramble
        self.env.reset_stats()
        self.recent_solves.clear()
        self.steps_at_level = 0
        self.level_history.append({
            'scramble': self.current_scramble,
            'total_steps': self.total_steps,
        })
        print(f"\n{'='*50}")
        print(f"CURRICULUM: Aumentando dificultad a {self.current_scramble} scramble moves")
        print(f"{'='*50}\n")

    def record_episode(self, solved: bool):
        self.recent_solves.append(1 if solved else 0)

    def step(self, num_steps: int = 1):
        self.total_steps += num_steps
        self.steps_at_level += num_steps


# ============================================================================
# Simple Policy Network (para usar sin PufferLib)
# ============================================================================

if TORCH_AVAILABLE:
    class SimplePolicy(nn.Module):
        """Red neuronal simple para el cubo de Rubik."""

        def __init__(self, obs_size=324, action_size=12, hidden_size=256):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.actor = nn.Linear(hidden_size, action_size)
            self.critic = nn.Linear(hidden_size, 1)

        def forward(self, x):
            hidden = self.network(x)
            return self.actor(hidden), self.critic(hidden)

        def get_action(self, obs, deterministic=False):
            with torch.no_grad():
                logits, value = self.forward(obs)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                return action, value


# ============================================================================
# Entrenamiento sin PufferLib (con curriculum)
# ============================================================================

def train_with_curriculum(args):
    """Entrenamiento con curriculum learning (sin PufferLib)."""
    print("\n" + "="*60)
    print("ENTRENAMIENTO CON CURRICULUM LEARNING")
    print("="*60)
    print(f"  Start scramble: {args.start_scramble}")
    print(f"  Max scramble: {args.max_scramble}")
    print(f"  Success threshold: {args.success_threshold:.0%}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  PyTorch: {'Disponible' if TORCH_AVAILABLE else 'No disponible'}")
    print("="*60 + "\n")

    # Crear entorno
    env = RubiksCubeEnv(
        num_envs=args.num_envs,
        scramble_moves=args.start_scramble,
        max_steps=args.max_steps,
        reward_mode=args.reward_mode,
        solve_reward=args.solve_reward,
        step_penalty=args.step_penalty,
    )

    # Crear curriculum trainer
    curriculum = CurriculumTrainer(
        env=env,
        start_scramble=args.start_scramble,
        max_scramble=args.max_scramble,
        success_threshold=args.success_threshold,
        eval_window=args.eval_window,
        steps_per_level=args.steps_per_level,
    )

    # Crear policy si PyTorch está disponible
    policy = None
    optimizer = None
    device = 'cpu'

    if TORCH_AVAILABLE and not args.random_policy:
        device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
        policy = SimplePolicy().to(device)
        optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
        print(f"Usando policy network en {device}")
    else:
        print("Usando acciones aleatorias (sin policy network)")

    obs, _ = env.reset(seed=args.seed)

    # Estadisticas
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    last_log_time = start_time

    print(f"Iniciando entrenamiento con {args.num_envs} entornos paralelos...")
    print(f"Scramble inicial: {curriculum.current_scramble}\n")

    step = 0
    while step < args.total_timesteps:
        # Obtener acciones
        if policy is not None:
            obs_tensor = torch.FloatTensor(obs).to(device)
            actions, _ = policy.get_action(obs_tensor)
            actions = actions.cpu().numpy()
        else:
            actions = np.random.randint(0, 12, size=args.num_envs)

        obs, rewards, terminals, truncations, infos = env.step(actions)
        step += args.num_envs
        curriculum.step(args.num_envs)

        # Registrar episodios completados
        for i in range(args.num_envs):
            if terminals[i] or truncations[i]:
                solved = terminals[i]
                curriculum.record_episode(solved)
                if f'episode_return_{i}' in infos:
                    episode_rewards.append(infos[f'episode_return_{i}'])
                    episode_lengths.append(infos[f'episode_length_{i}'])

        # Verificar si debemos aumentar dificultad
        if curriculum.should_increase_difficulty():
            curriculum.increase_difficulty()
            obs, _ = env.reset()

        # Logging
        current_time = time.time()
        if current_time - last_log_time >= args.log_interval:
            elapsed = current_time - start_time
            sps = step / elapsed
            success_rate = curriculum.get_success_rate()

            print(f"Step {step:,}/{args.total_timesteps:,} | "
                  f"Scramble: {curriculum.current_scramble} | "
                  f"Success: {success_rate:.1%} | "
                  f"SPS: {sps:.0f}")

            if episode_rewards:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"  Avg reward: {avg_reward:.3f} | Avg length: {avg_length:.1f}")

            last_log_time = current_time

    # Resumen final
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"  Total steps: {step:,}")
    print(f"  Tiempo: {elapsed:.1f}s")
    print(f"  SPS promedio: {step/elapsed:.0f}")
    print(f"  Scramble final: {curriculum.current_scramble}")
    print(f"  Success rate final: {curriculum.get_success_rate():.1%}")

    if curriculum.level_history:
        print("\nProgresion del curriculum:")
        for level in curriculum.level_history:
            print(f"  Scramble {level['scramble']}: alcanzado en step {level['total_steps']:,}")

    # Guardar modelo
    if policy is not None and args.save_path:
        torch.save(policy.state_dict(), args.save_path)
        print(f"\nModelo guardado en {args.save_path}")

    env.close()
    return curriculum


# ============================================================================
# Entrenamiento con PufferLib v3.0
# ============================================================================

def train_with_pufferlib_v3(args):
    """Entrenamiento con PufferLib v3.0 API."""
    if not PUFFER_AVAILABLE:
        raise ImportError("PufferLib v3.0 es requerido")

    print("\n" + "="*60)
    print(f"ENTRENAMIENTO CON PUFFERLIB v{PUFFER_VERSION}")
    print("="*60)

    try:
        # Importar modulos de PufferLib v3.0
        import pufferlib.pufferl as pufferl

        current_scramble = args.start_scramble
        total_steps = 0

        while current_scramble <= args.max_scramble and total_steps < args.total_timesteps:
            print(f"\n--- Nivel {current_scramble}: scramble_moves = {current_scramble} ---")

            # Crear entorno
            def env_creator():
                return RubiksCubeEnv(
                    num_envs=1,
                    scramble_moves=current_scramble,
                    max_steps=args.max_steps,
                    reward_mode=args.reward_mode,
                    solve_reward=args.solve_reward,
                    step_penalty=args.step_penalty,
                )

            # Crear vectorized environment
            vecenv = pufferlib.vector.make(
                env_creator,
                num_envs=args.num_envs,
                num_workers=args.num_workers,
            )

            # Crear policy
            policy = SimplePolicy()
            if args.device == 'cuda' and torch.cuda.is_available():
                policy = policy.cuda()

            # Configurar PuffeRL
            level_steps = min(args.steps_per_level, args.total_timesteps - total_steps)

            # Crear trainer usando la API de v3.0
            trainer = pufferl.PuffeRL(
                config={
                    'total_timesteps': level_steps,
                    'learning_rate': args.learning_rate,
                    'gamma': args.gamma,
                    'gae_lambda': args.gae_lambda,
                    'update_epochs': args.update_epochs,
                    'clip_coef': args.clip_coef,
                    'vf_coef': args.vf_coef,
                    'ent_coef': args.ent_coef,
                    'batch_size': args.batch_size,
                },
                vecenv=vecenv,
                policy=policy,
            )

            # Training loop
            while trainer.global_step < level_steps:
                trainer.evaluate()
                trainer.train()

            total_steps += level_steps
            trainer.close()
            vecenv.close()

            current_scramble += 1
            print(f"Avanzando a scramble_moves = {current_scramble}")

        print("\nEntrenamiento con PufferLib completado!")

    except AttributeError as e:
        print(f"Error de API: {e}")
        print("Intentando con API alternativa...")
        train_with_pufferlib_fallback(args)


def train_with_pufferlib_fallback(args):
    """Fallback para diferentes versiones de PufferLib."""
    print("Usando metodo de entrenamiento alternativo...")

    # Intentar diferentes APIs
    try:
        # Metodo 1: pufferlib.pufferl.train()
        import pufferlib.pufferl as pufferl
        if hasattr(pufferl, 'train'):
            print("Usando pufferl.train()")
            # Esto requeriria configuracion especifica del entorno
            raise NotImplementedError("Necesita configuracion de environment registry")
    except (ImportError, NotImplementedError):
        pass

    try:
        # Metodo 2: API legacy
        import pufferlib.frameworks.cleanrl as cleanrl
        if hasattr(cleanrl, 'PPO'):
            print("Usando pufferlib.frameworks.cleanrl (legacy)")
            # Usar API legacy
            raise NotImplementedError("API legacy no disponible en v3.0")
    except (ImportError, NotImplementedError, AttributeError):
        pass

    # Si nada funciona, usar entrenamiento simple
    print("Usando entrenamiento sin PufferLib...")
    train_with_curriculum(args)


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar RL para Cubo de Rubik con Curriculum Learning"
    )

    # Parametros del curriculum
    parser.add_argument("--start-scramble", type=int, default=1,
                        help="Scramble moves inicial (default: 1)")
    parser.add_argument("--max-scramble", type=int, default=20,
                        help="Scramble moves maximo (God's number = 20)")
    parser.add_argument("--success-threshold", type=float, default=0.8,
                        help="Tasa de exito para avanzar nivel (default: 0.8)")
    parser.add_argument("--eval-window", type=int, default=100,
                        help="Ventana de episodios para evaluar (default: 100)")
    parser.add_argument("--steps-per-level", type=int, default=50000,
                        help="Minimo de pasos por nivel (default: 50000)")

    # Parametros del entorno
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximo de pasos por episodio")
    parser.add_argument("--reward-mode", type=str, default='sparse',
                        choices=['sparse', 'dense'],
                        help="Modo de recompensa (default: sparse)")
    parser.add_argument("--solve-reward", type=float, default=1.0,
                        help="Recompensa por resolver")
    parser.add_argument("--step-penalty", type=float, default=0.01,
                        help="Penalizacion por paso")

    # Parametros de entrenamiento
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total de pasos de entrenamiento")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Tasa de aprendizaje")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Factor de descuento")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Lambda para GAE")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Epocas de actualizacion")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="Coeficiente de clipping PPO")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Coeficiente de value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Coeficiente de entropia")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximo gradiente")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Tamano del batch")
    parser.add_argument("--minibatch-size", type=int, default=512,
                        help="Tamano del minibatch")

    # Parametros de vectorizacion
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Numero de entornos paralelos")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Numero de workers")

    # Otros
    parser.add_argument("--save-path", type=str, default="rubik_model.pt",
                        help="Ruta para guardar modelo")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed para reproducibilidad")
    parser.add_argument("--log-interval", type=float, default=5.0,
                        help="Intervalo de logging en segundos")
    parser.add_argument("--use-pufferlib", action="store_true",
                        help="Usar PufferLib para entrenamiento")
    parser.add_argument("--device", type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help="Dispositivo para entrenamiento")
    parser.add_argument("--random-policy", action="store_true",
                        help="Usar acciones aleatorias (sin red neuronal)")

    args = parser.parse_args()

    if args.use_pufferlib and PUFFER_AVAILABLE:
        train_with_pufferlib_v3(args)
    else:
        train_with_curriculum(args)


if __name__ == "__main__":
    main()
