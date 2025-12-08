"""
Entrenamiento del Cubo de Rubik 2x2 (Pocket Cube)

El 2x2 es más simple que el 3x3:
- 24 stickers (vs 54)
- 9 movimientos (vs 12)
- God's number = 11 (vs 20)
- Perfecto para probar algoritmos rápidamente

Uso:
    # Entrenamiento rápido (2M steps, ~80% success en nivel 7)
    python train_2x2.py --total-timesteps 2_000_000

    # Entrenamiento completo (5M+ steps, ~95%+ success en nivel 11)
    python train_2x2.py --total-timesteps 10_000_000 --num-envs 512 --success-threshold 0.99

    # Con GPU
    python train_2x2.py --device cuda --num-envs 1024

    # Ajustar curriculum
    python train_2x2.py --success-threshold 0.99 --min-steps-per-level 50000

    # RESUME TRAINING: Continuar desde checkpoint anterior
    python train_2x2.py --load-path rubik_2x2.pt --save-path rubik_2x2_v2.pt
"""

import argparse
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.set_num_threads(1)

# Import 2x2 C extension
try:
    import rubik2x2_c
    C_AVAILABLE = True
    print(f">>> CUBO 2x2 DISPONIBLE - {rubik2x2_c.NUM_ACTIONS} acciones, {rubik2x2_c.OBS_SIZE} obs <<<")
except ImportError as e:
    C_AVAILABLE = False
    print(f">>> CUBO 2x2 NO DISPONIBLE: {e} <<<")
    print(">>> Run: python setup.py build_ext --inplace <<<")


# ============================================================================
# Policy Network (más pequeña para 2x2)
# ============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


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


class Policy2x2(nn.Module):
    """Red más pequeña para el cubo 2x2."""

    def __init__(self, obs_size=144, action_size=9, hidden_size=256, num_blocks=2):
        super().__init__()

        self.input_fc = layer_init(nn.Linear(obs_size, hidden_size))
        self.input_ln = nn.LayerNorm(hidden_size)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        self.actor_fc = layer_init(nn.Linear(hidden_size, hidden_size // 2))
        self.action_head = layer_init(nn.Linear(hidden_size // 2, action_size), std=0.01)
        self.critic_fc = layer_init(nn.Linear(hidden_size, hidden_size // 2))
        self.value_head = layer_init(nn.Linear(hidden_size // 2, 1), std=1.0)

    def forward(self, x):
        x = F.relu(self.input_ln(self.input_fc(x)))
        for block in self.residual_blocks:
            x = block(x)
        actor = F.relu(self.actor_fc(x))
        logits = self.action_head(actor)
        critic = F.relu(self.critic_fc(x))
        value = self.value_head(critic)
        return logits, value.squeeze(-1)

    def get_action_and_value(self, obs, action=None, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    def __init__(
        self,
        env,
        policy,
        device='cpu',
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        num_steps=128,
        num_minibatches=4,
        update_epochs=4,
        start_scramble=1,
        max_scramble=11,  # God's number for 2x2
        success_threshold=0.98,  # Aumentado de 0.95 a 0.98 para mejor maestría
        curriculum_window=200,   # Más muestras para decidir avance
        min_steps_per_level=40000,  # Más pasos por nivel antes de avanzar
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.num_envs = env.num_envs

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

        self.current_scramble = start_scramble
        self.max_scramble = max_scramble
        self.success_threshold = success_threshold
        self.curriculum_window = curriculum_window
        self.min_steps_per_level = min_steps_per_level
        self.recent_solves = deque(maxlen=curriculum_window)
        self.steps_at_level = 0

        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        obs_size = rubik2x2_c.OBS_SIZE
        self.obs_buffer = torch.zeros((num_steps, self.num_envs, obs_size), device=device)
        self.actions_buffer = torch.zeros((num_steps, self.num_envs), dtype=torch.long, device=device)
        self.logprobs_buffer = torch.zeros((num_steps, self.num_envs), device=device)
        self.rewards_buffer = torch.zeros((num_steps, self.num_envs), device=device)
        self.dones_buffer = torch.zeros((num_steps, self.num_envs), device=device)
        self.values_buffer = torch.zeros((num_steps, self.num_envs), device=device)

        self.global_step = 0
        self.solve_count = 0
        self.episode_count = 0

    def get_success_rate(self):
        return sum(self.recent_solves) / len(self.recent_solves) if self.recent_solves else 0.0

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

    def collect_rollout(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)

        for step in range(self.num_steps):
            with torch.no_grad():
                action, logprob, _, value = self.policy.get_action_and_value(obs)

            self.obs_buffer[step] = obs
            self.actions_buffer[step] = action
            self.logprobs_buffer[step] = logprob
            self.values_buffer[step] = value

            next_obs, rewards, terminals, truncations, _ = self.env.step(
                action.cpu().numpy().astype(np.int32)
            )

            dones = np.logical_or(terminals, truncations)
            self.rewards_buffer[step] = torch.FloatTensor(rewards).to(self.device)
            self.dones_buffer[step] = torch.FloatTensor(dones).to(self.device)

            for i, done in enumerate(dones):
                if done:
                    self.recent_solves.append(1 if terminals[i] else 0)
                    self.episode_count += 1
                    if terminals[i]:
                        self.solve_count += 1

            obs = torch.FloatTensor(next_obs).to(self.device)

        with torch.no_grad():
            next_value = self.policy.get_value(obs)
            advantages = self._compute_gae(next_value)
            returns = advantages + self.values_buffer

        return obs, advantages, returns

    def _compute_gae(self, next_value):
        advantages = torch.zeros_like(self.rewards_buffer)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            next_non_terminal = 1.0 - self.dones_buffer[t]
            next_values = next_value if t == self.num_steps - 1 else self.values_buffer[t + 1]
            delta = self.rewards_buffer[t] + self.gamma * next_values * next_non_terminal - self.values_buffer[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        return advantages

    def update(self, advantages, returns):
        obs_size = rubik2x2_c.OBS_SIZE
        b_obs = self.obs_buffer.reshape(-1, obs_size)
        b_actions = self.actions_buffer.reshape(-1)
        b_logprobs = self.logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        clipfracs = []
        for _ in range(self.update_epochs):
            indices = torch.randperm(self.batch_size, device=self.device)

            for start in range(0, self.batch_size, self.minibatch_size):
                mb_indices = indices[start:start + self.minibatch_size]
                mb_obs = b_obs[mb_indices]

                _, new_logprob, entropy, new_value = self.policy.get_action_and_value(
                    mb_obs, b_actions[mb_indices]
                )

                logratio = new_logprob - b_logprobs[mb_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_indices]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {'pg_loss': pg_loss.item(), 'v_loss': v_loss.item(),
                'entropy': entropy_loss.item(), 'clipfrac': np.mean(clipfracs)}

    def train(self, total_timesteps, log_interval=1):
        print(f"\n{'='*60}")
        print("ENTRENAMIENTO CUBO 2x2")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Max scramble: {self.max_scramble} (God's number)")
        print(f"\n  Curriculum:")
        print(f"    Success threshold: {self.success_threshold:.1%}")
        print(f"    Window: {self.curriculum_window} episodes")
        print(f"    Min steps/level: {self.min_steps_per_level:,}")
        print(f"{'='*60}\n")

        num_updates = total_timesteps // self.batch_size
        start_time = time.time()
        obs, _ = self.env.reset()

        for update in range(1, num_updates + 1):
            obs, advantages, returns = self.collect_rollout(obs)
            losses = self.update(advantages, returns)

            self.global_step += self.batch_size
            self.steps_at_level += self.batch_size

            if self.should_increase_difficulty():
                self.increase_difficulty()

            if update % log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed
                print(f"Update {update}/{num_updates} | Step {self.global_step:,} | "
                      f"Scramble {self.current_scramble}/{self.max_scramble} | "
                      f"Success {self.get_success_rate():.1%} | SPS {sps:,.0f}")
                print(f"  pg={losses['pg_loss']:.4f} v={losses['v_loss']:.4f} "
                      f"ent={losses['entropy']:.4f} clip={losses['clipfrac']:.3f}")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETADO - {self.global_step:,} steps en {elapsed:.1f}s ({self.global_step/elapsed:,.0f} SPS)")
        print(f"Success rate: {self.get_success_rate():.1%} | Solves: {self.solve_count:,}/{self.episode_count:,}")
        print(f"Scramble level: {self.current_scramble}/{self.max_scramble}")
        print(f"{'='*60}")


# ============================================================================
# Wrapper for batch environment
# ============================================================================

class Rubik2x2BatchEnv:
    """Wrapper around C batch environment."""

    def __init__(self, num_envs, scramble_moves, max_steps, seed=42):
        self._env = rubik2x2_c.Rubik2x2BatchEnv(
            num_envs=num_envs,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=1.0,
            step_penalty=0.01,
            reward_mode=0,
            seed=seed,
        )
        self.num_envs = num_envs
        self._scramble_moves = scramble_moves

    @property
    def scramble_moves(self):
        return self._scramble_moves

    @scramble_moves.setter
    def scramble_moves(self, value):
        self._scramble_moves = value
        self._env.scramble_moves = value

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        return self._env.step(actions)

    def stagger(self):
        """Apply staggered resets to desynchronize environments (paper 2511.21011)."""
        if hasattr(self._env, 'stagger'):
            self._env.stagger()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenar Cubo 2x2 con PPO")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--scramble-moves", type=int, default=1)
    parser.add_argument("--max-scramble", type=int, default=11)  # God's number
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
                        help="Total training steps (default: 5M for good performance)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="rubik_2x2.pt")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--log-interval", type=int, default=1)

    # Curriculum parameters
    parser.add_argument("--success-threshold", type=float, default=0.98,
                        help="Success rate required to advance curriculum (default: 0.98)")
    parser.add_argument("--curriculum-window", type=int, default=200,
                        help="Number of episodes to average for curriculum decision (default: 200)")
    parser.add_argument("--min-steps-per-level", type=int, default=40000,
                        help="Minimum steps before advancing curriculum (default: 40000)")

    args = parser.parse_args()

    if not C_AVAILABLE:
        print("ERROR: Módulo 2x2 no disponible.")
        print("Ejecuta: python setup.py build_ext --inplace")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU")
        args.device = "cpu"

    # Create environment
    print(f"Creando {args.num_envs} entornos 2x2...")
    env = Rubik2x2BatchEnv(
        num_envs=args.num_envs,
        scramble_moves=args.scramble_moves,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Apply staggered resets (paper 2511.21011)
    # This desynchronizes environment resets to avoid cyclic non-stationarity
    print("Aplicando Staggered Resets...")
    env.reset()    # First reset all to step 0
    env.stagger()  # Then distribute step counts uniformly
    print(f"Entorno creado: {env.num_envs} agentes (staggered)")

    # Create policy
    policy = Policy2x2(
        obs_size=rubik2x2_c.OBS_SIZE,
        action_size=rubik2x2_c.NUM_ACTIONS,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
    )

    # Resume training: load checkpoint if provided
    start_scramble = args.scramble_moves
    if args.load_path:
        print(f"\n>>> CARGANDO CHECKPOINT: {args.load_path} <<<")
        checkpoint = torch.load(args.load_path, map_location=args.device)

        # Get saved architecture params to recreate policy correctly
        saved_args = checkpoint.get('args', {})
        saved_hidden = saved_args.get('hidden_size', args.hidden_size)
        saved_blocks = saved_args.get('num_blocks', args.num_blocks)

        # Recreate policy with correct architecture if different
        if saved_hidden != args.hidden_size or saved_blocks != args.num_blocks:
            print(f">>> Usando arquitectura del checkpoint: hidden={saved_hidden}, blocks={saved_blocks} <<<")
            policy = Policy2x2(
                obs_size=rubik2x2_c.OBS_SIZE,
                action_size=rubik2x2_c.NUM_ACTIONS,
                hidden_size=saved_hidden,
                num_blocks=saved_blocks,
            )

        # Load policy weights (handle torch.compile prefix)
        state_dict = checkpoint['policy_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        policy.load_state_dict(new_state_dict)

        # Restore curriculum level
        if 'scramble_level' in checkpoint:
            start_scramble = checkpoint['scramble_level']
            env.scramble_moves = start_scramble
            print(f">>> RETOMANDO DESDE NIVEL: {start_scramble}/{args.max_scramble} <<<\n")

    # Try to compile with torch 2.0+
    if int(torch.__version__.split(".")[0]) >= 2:
        print(">>> Compilando modelo... <<<")
        policy = torch.compile(policy)

    print(f"Policy: {sum(p.numel() for p in policy.parameters()):,} parametros")

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        device=args.device,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        start_scramble=start_scramble,  # Use loaded level or default
        max_scramble=args.max_scramble,
        success_threshold=args.success_threshold,
        curriculum_window=args.curriculum_window,
        min_steps_per_level=args.min_steps_per_level,
    )

    # Train
    trainer.train(args.total_timesteps, args.log_interval)

    # Save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'scramble_level': trainer.current_scramble,
        'args': vars(args),
    }, args.save_path)
    print(f"Modelo guardado en {args.save_path}")


if __name__ == "__main__":
    main()
