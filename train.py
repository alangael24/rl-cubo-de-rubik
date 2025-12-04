"""
Script de entrenamiento para el Cubo de Rubik usando PufferLib.

Uso:
    python train.py
    python train.py --scramble-moves 10 --max-steps 100
    python train.py --total-timesteps 1000000
"""

import argparse
import os
import numpy as np

try:
    import pufferlib
    import pufferlib.vector
    import pufferlib.frameworks.cleanrl
    PUFFER_AVAILABLE = True
except ImportError:
    PUFFER_AVAILABLE = False
    print("PufferLib no esta instalado. Instalalo con: pip install pufferlib")

from rubik_env import RubiksCubeEnv, make_env


def train_with_pufferlib(args):
    """Entrena usando PufferLib."""
    if not PUFFER_AVAILABLE:
        raise ImportError("PufferLib es requerido para entrenar")

    print("Configurando entrenamiento con PufferLib...")
    print(f"  - Scramble moves: {args.scramble_moves}")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Total timesteps: {args.total_timesteps}")
    print(f"  - Num envs: {args.num_envs}")

    # Crear vectorizador de entornos
    env_creator = make_env(
        scramble_moves=args.scramble_moves,
        max_steps=args.max_steps,
        reward_scale=args.reward_scale,
        step_penalty=args.step_penalty,
    )

    # Configurar vectorizacion
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        backend=pufferlib.vector.Multiprocessing,
    )

    # Configurar el trainer de CleanRL
    config = pufferlib.frameworks.cleanrl.Config(
        total_timesteps=args.total_timesteps,
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

    # Entrenar
    print("\nIniciando entrenamiento...")
    trainer = pufferlib.frameworks.cleanrl.PPO(
        vecenv=vecenv,
        config=config,
    )

    trainer.train()

    # Guardar modelo
    if args.save_path:
        print(f"\nGuardando modelo en {args.save_path}")
        trainer.save(args.save_path)

    vecenv.close()
    print("\nEntrenamiento completado!")


def train_simple(args):
    """Entrenamiento simple sin PufferLib (para testing)."""
    print("Ejecutando entrenamiento simple (sin PufferLib)...")
    print("Esto es solo para probar que el entorno funciona.\n")

    env = RubiksCubeEnv(
        num_envs=args.num_envs,
        scramble_moves=args.scramble_moves,
        max_steps=args.max_steps,
    )

    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.single_action_space}")

    total_rewards = np.zeros(args.num_envs)
    episodes_completed = 0
    solved_count = 0

    print(f"\nEjecutando {args.total_timesteps} pasos...")

    for step in range(args.total_timesteps):
        # Acciones aleatorias
        actions = np.random.randint(0, 12, size=args.num_envs)
        obs, rewards, terminals, truncations, infos = env.step(actions)

        total_rewards += rewards

        # Contar episodios completados
        for i in range(args.num_envs):
            if terminals[i] or truncations[i]:
                episodes_completed += 1
                if terminals[i]:  # Resuelto
                    solved_count += 1

        if (step + 1) % 1000 == 0:
            avg_reward = np.mean(total_rewards) / (step + 1) * args.max_steps
            print(f"Step {step + 1}: avg_reward={avg_reward:.4f}, "
                  f"episodes={episodes_completed}, solved={solved_count}")

    print(f"\n=== Resultados ===")
    print(f"Total pasos: {args.total_timesteps}")
    print(f"Episodios completados: {episodes_completed}")
    print(f"Cubos resueltos: {solved_count}")
    print(f"Tasa de resolucion: {solved_count / max(1, episodes_completed) * 100:.2f}%")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Entrenar RL para Cubo de Rubik")

    # Parametros del entorno
    parser.add_argument("--scramble-moves", type=int, default=20,
                        help="Numero de movimientos para mezclar el cubo")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximo de pasos por episodio")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Escala de recompensa")
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
                        help="Epocas de actualizacion por batch")
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
                        help="Reducir learning rate linealmente")

    # Parametros de vectorizacion
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Numero de entornos paralelos")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Numero de workers para multiprocessing")

    # Otros
    parser.add_argument("--save-path", type=str, default="rubik_model.pt",
                        help="Ruta para guardar el modelo")
    parser.add_argument("--simple", action="store_true",
                        help="Usar entrenamiento simple sin PufferLib")

    args = parser.parse_args()

    if args.simple or not PUFFER_AVAILABLE:
        train_simple(args)
    else:
        train_with_pufferlib(args)


if __name__ == "__main__":
    main()
