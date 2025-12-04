"""
Script de entrenamiento para el Cubo de Rubik usando Curriculum Learning.

Basado en recomendaciones de papers academicos:
- DeepCubeA (McAleer et al., 2019)
- Autodidactic Iteration (ADI)

Curriculum Learning:
1. Empezar con cubos mezclados con 1 movimiento
2. Cuando success_rate > threshold, aumentar dificultad
3. Progresar hasta scramble_moves = 20 (God's number)

Uso:
    python train.py
    python train.py --max-scramble 10 --success-threshold 0.8
"""

import argparse
import os
import time
from collections import deque
import numpy as np

try:
    import pufferlib
    import pufferlib.vector
    import pufferlib.frameworks.cleanrl
    PUFFER_AVAILABLE = True
except ImportError:
    PUFFER_AVAILABLE = False
    print("PufferLib no esta instalado. Usando entrenamiento simple.")

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
        """
        Args:
            env: Entorno de Rubik's Cube
            start_scramble: Scramble inicial (default: 1)
            max_scramble: Maximo scramble (God's number = 20)
            success_threshold: Tasa de exito para avanzar nivel
            eval_window: Ventana de episodios para evaluar exito
            steps_per_level: Minimo de pasos antes de poder avanzar
        """
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
        """Retorna tasa de exito de la ventana reciente."""
        if len(self.recent_solves) == 0:
            return 0.0
        return sum(self.recent_solves) / len(self.recent_solves)

    def should_increase_difficulty(self) -> bool:
        """Determina si debemos aumentar la dificultad."""
        if self.current_scramble >= self.max_scramble:
            return False
        if self.steps_at_level < self.steps_per_level:
            return False
        if len(self.recent_solves) < self.eval_window:
            return False
        return self.get_success_rate() >= self.success_threshold

    def increase_difficulty(self):
        """Aumenta la dificultad (mas scramble moves)."""
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
        """Registra resultado de un episodio."""
        self.recent_solves.append(1 if solved else 0)

    def step(self, num_steps: int = 1):
        """Actualiza contadores de pasos."""
        self.total_steps += num_steps
        self.steps_at_level += num_steps


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
        # Acciones aleatorias (o de policy si tienes una)
        # En un entrenamiento real, aqui iria tu policy network
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

    env.close()
    return curriculum


def train_with_pufferlib(args):
    """Entrenamiento completo con PufferLib y curriculum."""
    if not PUFFER_AVAILABLE:
        raise ImportError("PufferLib es requerido")

    print("\n" + "="*60)
    print("ENTRENAMIENTO CON PUFFERLIB + CURRICULUM")
    print("="*60)

    # Para PufferLib, el curriculum se maneja diferente
    # porque el vecenv no permite cambiar scramble_moves facilmente
    # Una solucion es entrenar por niveles

    current_scramble = args.start_scramble
    total_steps = 0

    while current_scramble <= args.max_scramble and total_steps < args.total_timesteps:
        print(f"\n--- Nivel {current_scramble}: scramble_moves = {current_scramble} ---")

        # Crear entorno para este nivel
        env_creator = make_env(
            scramble_moves=current_scramble,
            max_steps=args.max_steps,
            reward_mode=args.reward_mode,
            solve_reward=args.solve_reward,
            step_penalty=args.step_penalty,
        )

        vecenv = pufferlib.vector.make(
            env_creator,
            num_envs=args.num_envs,
            num_workers=args.num_workers,
            backend=pufferlib.vector.Multiprocessing,
        )

        # Pasos para este nivel
        level_steps = min(args.steps_per_level, args.total_timesteps - total_steps)

        config = pufferlib.frameworks.cleanrl.Config(
            total_timesteps=level_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            update_epochs=args.update_epochs,
            clip_coef=args.clip_coef,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
            minibatch_size=args.minibatch_size,
            anneal_lr=args.anneal_lr,
        )

        trainer = pufferlib.frameworks.cleanrl.PPO(
            vecenv=vecenv,
            config=config,
        )

        trainer.train()
        total_steps += level_steps

        # Evaluar y decidir si avanzar
        # (En una implementacion real, evaluarias el success rate)
        vecenv.close()

        current_scramble += 1
        print(f"Avanzando a scramble_moves = {current_scramble}")

    # Guardar modelo final
    if args.save_path:
        print(f"\nGuardando modelo en {args.save_path}")
        trainer.save(args.save_path)

    print("\nEntrenamiento completado!")


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
    parser.add_argument("--anneal-lr", action="store_true", default=True,
                        help="Reducir learning rate")

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

    args = parser.parse_args()

    if args.use_pufferlib and PUFFER_AVAILABLE:
        train_with_pufferlib(args)
    else:
        train_with_curriculum(args)


if __name__ == "__main__":
    main()
