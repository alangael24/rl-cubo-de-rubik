"""
Script de entrenamiento para el Cubo de Rubik con ResNet y PPO.

Caracteristicas:
- Backend C de alto rendimiento (~5M+ steps/segundo)
- ResNet como policy network (recomendado para puzzles)
- PPO (Proximal Policy Optimization)
- Curriculum Learning (ADI - Autodidactic Iteration)

Uso:
    python train.py                          # Entrenar con defaults
    python train.py --num-envs 256           # Mas entornos paralelos
    python train.py --max-scramble 5         # Limitar dificultad
    python train.py --device cuda            # Usar GPU
"""

import argparse
import os
import time
from collections import deque
import numpy as np

# ============================================================================
# Importar backend C
# ============================================================================

from rubik_env_c import RubiksCubeBatchEnvC as RubiksCubeEnv
C_BACKEND = True
print(">>> USANDO BACKEND C (OPTIMIZADO) <<<")

# Symmetry data augmentation
try:
    from symmetry import apply_random_symmetry_batch, NUM_SYMMETRIES
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    print("Warning: symmetry module not found. Data augmentation disabled.")

# ============================================================================
# PyTorch
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch es requerido para entrenar.")
    print("Instala con: pip install torch")
    exit(1)


# ============================================================================
# ResNet Policy Network
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
        x = F.relu(x + residual)  # Skip connection
        return x


class ResNetPolicy(nn.Module):
    """
    ResNet Policy para el Cubo de Rubik.

    Arquitectura basada en DeepCubeA:
    - Input: One-hot encoding (324 = 54 stickers * 6 colores)
    - Varios bloques residuales
    - Separate heads para actor (policy) y critic (value)
    """

    def __init__(
        self,
        obs_size: int = 324,
        action_size: int = 12,
        hidden_size: int = 512,
        num_residual_blocks: int = 4,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.action_size = action_size

        # Input projection
        self.input_fc = nn.Linear(obs_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_residual_blocks)
        ])

        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_out = nn.Linear(hidden_size // 2, action_size)

        # Critic head (value)
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_out = nn.Linear(hidden_size // 2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Inicializacion ortogonal (mejor para RL)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Escalar output layers
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)

    def forward(self, x):
        """Forward pass completo."""
        # Input projection
        x = F.relu(self.input_ln(self.input_fc(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Actor (policy logits)
        actor = F.relu(self.actor_fc(x))
        logits = self.actor_out(actor)

        # Critic (value)
        critic = F.relu(self.critic_fc(x))
        value = self.critic_out(critic)

        return logits, value.squeeze(-1)

    def get_action_and_value(self, obs, action=None, deterministic=False):
        """
        Obtiene accion, log_prob, entropia y valor.

        Args:
            obs: Observaciones [batch, obs_size]
            action: Acciones previas (para calcular log_prob)
            deterministic: Si True, toma la accion greedy

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, obs):
        """Solo obtiene el valor (para GAE)."""
        _, value = self.forward(obs)
        return value


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization con Curriculum Learning.
    """

    def __init__(
        self,
        env,
        policy: ResNetPolicy,
        device: str = 'cpu',
        # PPO hyperparameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Training settings
        num_steps: int = 128,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        # Curriculum
        start_scramble: int = 1,
        max_scramble: int = 20,
        success_threshold: float = 0.8,
        curriculum_window: int = 100,
        min_steps_per_level: int = 50000,
        # Data augmentation
        use_symmetry_aug: bool = True,
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.num_envs = env.num_envs

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs

        # Data augmentation
        self.use_symmetry_aug = use_symmetry_aug and SYMMETRY_AVAILABLE

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

        # Curriculum learning
        self.current_scramble = start_scramble
        self.max_scramble = max_scramble
        self.success_threshold = success_threshold
        self.curriculum_window = curriculum_window
        self.min_steps_per_level = min_steps_per_level
        self.recent_solves = deque(maxlen=curriculum_window)
        self.steps_at_level = 0

        # Batch size
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        # Rollout buffers
        self.obs_buffer = torch.zeros((num_steps, self.num_envs, 324), device=device)
        self.actions_buffer = torch.zeros((num_steps, self.num_envs), dtype=torch.long, device=device)
        self.logprobs_buffer = torch.zeros((num_steps, self.num_envs), device=device)
        self.rewards_buffer = torch.zeros((num_steps, self.num_envs), device=device)
        self.dones_buffer = torch.zeros((num_steps, self.num_envs), device=device)
        self.values_buffer = torch.zeros((num_steps, self.num_envs), device=device)

        # Statistics
        self.global_step = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.solve_count = 0
        self.episode_count = 0

    def get_success_rate(self):
        if len(self.recent_solves) == 0:
            return 0.0
        return sum(self.recent_solves) / len(self.recent_solves)

    def should_increase_difficulty(self):
        if self.current_scramble >= self.max_scramble:
            return False
        if self.steps_at_level < self.min_steps_per_level:
            return False
        if len(self.recent_solves) < self.curriculum_window:
            return False
        return self.get_success_rate() >= self.success_threshold

    def increase_difficulty(self):
        self.current_scramble += 1
        self.env.scramble_moves = self.current_scramble
        self.recent_solves.clear()
        self.steps_at_level = 0
        print(f"\n{'='*50}")
        print(f"CURRICULUM: Nivel {self.current_scramble} scramble moves")
        print(f"{'='*50}\n")

    def collect_rollout(self):
        """Recolecta un rollout de experiencias."""
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        for step in range(self.num_steps):
            with torch.no_grad():
                action, logprob, _, value = self.policy.get_action_and_value(obs)

            # Store
            self.obs_buffer[step] = obs
            self.actions_buffer[step] = action
            self.logprobs_buffer[step] = logprob
            self.values_buffer[step] = value

            # Step environment
            next_obs, rewards, terminals, truncations, infos = self.env.step(
                action.cpu().numpy()
            )

            dones = np.logical_or(terminals, truncations)

            self.rewards_buffer[step] = torch.FloatTensor(rewards).to(self.device)
            self.dones_buffer[step] = torch.FloatTensor(dones).to(self.device)

            # Track episodes
            for i, done in enumerate(dones):
                if done:
                    solved = terminals[i]
                    self.recent_solves.append(1 if solved else 0)
                    self.episode_count += 1
                    if solved:
                        self.solve_count += 1

            obs = torch.FloatTensor(next_obs).to(self.device)

        # Compute returns with GAE
        with torch.no_grad():
            next_value = self.policy.get_value(obs)
            advantages = self._compute_gae(next_value)
            returns = advantages + self.values_buffer

        return advantages, returns

    def _compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards_buffer)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones_buffer[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones_buffer[t]
                next_values = self.values_buffer[t + 1]

            delta = (self.rewards_buffer[t] +
                     self.gamma * next_values * next_non_terminal -
                     self.values_buffer[t])
            advantages[t] = last_gae = (delta +
                                        self.gamma * self.gae_lambda *
                                        next_non_terminal * last_gae)

        return advantages

    def update(self, advantages, returns):
        """PPO update step with optional symmetry data augmentation."""
        # Flatten batches
        b_obs = self.obs_buffer.reshape(-1, 324)
        b_actions = self.actions_buffer.reshape(-1)
        b_logprobs = self.logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values_buffer.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update epochs
        clipfracs = []
        for epoch in range(self.update_epochs):
            # Shuffle indices
            indices = torch.randperm(self.batch_size, device=self.device)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch observations
                mb_obs = b_obs[mb_indices]

                # Apply symmetry augmentation if enabled
                # This forces the network to learn rotation-invariant features
                if self.use_symmetry_aug:
                    mb_obs = apply_random_symmetry_batch(mb_obs)

                _, new_logprob, entropy, new_value = self.policy.get_action_and_value(
                    mb_obs, b_actions[mb_indices]
                )

                # Policy loss
                logratio = new_logprob - b_logprobs[mb_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_indices]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            'pg_loss': pg_loss.item(),
            'v_loss': v_loss.item(),
            'entropy': entropy_loss.item(),
            'clipfrac': np.mean(clipfracs),
        }

    def train(self, total_timesteps: int, log_interval: int = 10):
        """Loop de entrenamiento principal."""
        print(f"\n{'='*60}")
        print("INICIANDO ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"  Backend: {'C (optimizado)' if C_BACKEND else 'Python (lento)'}")
        print(f"  Device: {self.device}")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Scramble inicial: {self.current_scramble}")
        if self.use_symmetry_aug:
            print(f"  Symmetry augmentation: ENABLED (24x data multiplier)")
        else:
            print(f"  Symmetry augmentation: disabled")
        print(f"{'='*60}\n")

        # Set initial scramble
        self.env.scramble_moves = self.current_scramble

        num_updates = total_timesteps // self.batch_size
        start_time = time.time()

        for update in range(1, num_updates + 1):
            # Collect rollout
            advantages, returns = self.collect_rollout()

            # PPO update
            losses = self.update(advantages, returns)

            # Update stats
            self.global_step += self.batch_size
            self.steps_at_level += self.batch_size

            # Check curriculum
            if self.should_increase_difficulty():
                self.increase_difficulty()

            # Logging
            if update % log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed
                success_rate = self.get_success_rate()

                print(f"Update {update}/{num_updates} | "
                      f"Step {self.global_step:,} | "
                      f"Scramble {self.current_scramble} | "
                      f"Success {success_rate:.1%} | "
                      f"SPS {sps:,.0f}")
                print(f"  pg_loss={losses['pg_loss']:.4f} "
                      f"v_loss={losses['v_loss']:.4f} "
                      f"entropy={losses['entropy']:.4f} "
                      f"clipfrac={losses['clipfrac']:.3f}")

        # Final stats
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"  Total steps: {self.global_step:,}")
        print(f"  Tiempo: {elapsed:.1f}s")
        print(f"  SPS promedio: {self.global_step/elapsed:,.0f}")
        print(f"  Scramble final: {self.current_scramble}")
        print(f"  Success rate: {self.get_success_rate():.1%}")
        print(f"  Episodes: {self.episode_count:,}")
        print(f"  Solves: {self.solve_count:,}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenar Cubo de Rubik con ResNet + PPO")

    # Environment
    parser.add_argument("--num-envs", type=int, default=256,
                        help="Numero de entornos paralelos")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps por episodio")

    # Curriculum
    parser.add_argument("--start-scramble", type=int, default=1,
                        help="Scramble inicial")
    parser.add_argument("--max-scramble", type=int, default=20,
                        help="Scramble maximo")
    parser.add_argument("--success-threshold", type=float, default=0.8,
                        help="Tasa de exito para avanzar")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="Total de timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Steps por rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Numero de minibatches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Epochs por update")

    # Network
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Tamano de capas ocultas")
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="Numero de bloques residuales")

    # Data augmentation
    parser.add_argument("--use-symmetry-aug", action="store_true", default=True,
                        help="Enable symmetry data augmentation (24x effective data)")
    parser.add_argument("--no-symmetry-aug", action="store_false", dest="use_symmetry_aug",
                        help="Disable symmetry data augmentation")

    # Other
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-path", type=str, default="rubik_resnet.pt",
                        help="Path para guardar modelo")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Intervalo de logging")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU")
        args.device = "cpu"

    # Create environment
    if C_BACKEND:
        env = RubiksCubeEnv(
            num_envs=args.num_envs,
            scramble_moves=args.start_scramble,
            max_steps=args.max_steps,
            reward_mode='sparse',
        )
    else:
        env = RubiksCubeEnv(
            num_envs=args.num_envs,
            scramble_moves=args.start_scramble,
            max_steps=args.max_steps,
            reward_mode='sparse',
        )

    # Create policy
    policy = ResNetPolicy(
        obs_size=324,
        action_size=12,
        hidden_size=args.hidden_size,
        num_residual_blocks=args.num_blocks,
    )

    print(f"\nResNet Policy:")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Residual blocks: {args.num_blocks}")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        device=args.device,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        start_scramble=args.start_scramble,
        max_scramble=args.max_scramble,
        success_threshold=args.success_threshold,
        use_symmetry_aug=args.use_symmetry_aug,
    )

    # Train
    trainer.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
    )

    # Save model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'scramble_level': trainer.current_scramble,
        'args': vars(args),
    }, args.save_path)
    print(f"\nModelo guardado en {args.save_path}")


if __name__ == "__main__":
    main()
