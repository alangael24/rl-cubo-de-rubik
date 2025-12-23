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
                 reward_mode='sparse', buf=None, seed=0, **kwargs):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(324,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(12)
        self.num_agents = num_envs

        # Inicializar buffers via parent class
        super().__init__(buf)

        # Crear entorno C
        self._c_env = rubik_c.RubikBatchEnv(
            num_envs=num_envs,
            scramble_moves=scramble_moves,
            max_steps=max_steps,
            solve_reward=1.0,
            step_penalty=0.01,
            reward_mode=0 if reward_mode == 'sparse' else 1,
            seed=seed,
        )

        # Buffers uint8 para terminals/truncations
        self._terms_u8 = np.zeros(num_envs, dtype=np.uint8)
        self._truncs_u8 = np.zeros(num_envs, dtype=np.uint8)

        # ZERO-COPY: C escribe directo a estos buffers
        self._c_env.set_buffers(self.observations, self.rewards, self._terms_u8, self._truncs_u8)

        # Stats para curriculum learning
        self._episode_solved = 0
        self._episode_count = 0

    @property
    def emulated(self):
        return None

    def reset(self, seed=None):
        self._c_env.reset()
        return self.observations, []

    def step(self, actions):
        # Llamar a C - escribe DIRECTO a buffers
        self._c_env.step(actions)

        # Conversion uint8 -> bool (necesario para PufferLib)
        self.terminals[:] = self._terms_u8
        self.truncations[:] = self._truncs_u8

        # Track solved episodes para curriculum
        self._episode_solved += self.terminals.sum()
        self._episode_count += (self.terminals | self.truncations).sum()

        return self.observations, self.rewards, self.terminals, self.truncations, []

    def get_solve_rate(self):
        if self._episode_count == 0:
            return 0.0
        return float(self._episode_solved) / float(self._episode_count)

    def reset_stats(self):
        self._episode_solved = 0
        self._episode_count = 0

    def set_scramble(self, n):
        self._c_env.scramble_moves = n
        print(f"  >>> CURRICULUM: scramble_moves = {n} <<<")

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


def create_cube_rotations():
    """
    Crea las 24 permutaciones de stickers para rotaciones del cubo completo.
    Observation: 54 stickers x 6 colores (one-hot) = 324 valores.
    Faces: F(0-8), B(9-17), U(18-26), D(27-35), L(36-44), R(45-53)
    """
    def rotate_face_cw(stickers):
        """Rota los 9 stickers de una cara en sentido horario."""
        return [stickers[6], stickers[3], stickers[0],
                stickers[7], stickers[4], stickers[1],
                stickers[8], stickers[5], stickers[2]]

    def rotate_face_ccw(stickers):
        """Rota los 9 stickers de una cara en sentido antihorario."""
        return [stickers[2], stickers[5], stickers[8],
                stickers[1], stickers[4], stickers[7],
                stickers[0], stickers[3], stickers[6]]

    def rotate_face_180(stickers):
        return rotate_face_cw(rotate_face_cw(stickers))

    def get_face(perm, face_idx):
        return perm[face_idx*9:(face_idx+1)*9]

    def set_face(perm, face_idx, stickers):
        perm[face_idx*9:(face_idx+1)*9] = stickers

    # Identidad
    identity = list(range(54))

    # Rotación X (alrededor del eje R-L): F->U->B->D->F
    def rot_x(perm):
        new_perm = perm.copy()
        # F -> U, U -> B (rotado 180), B -> D, D -> F
        set_face(new_perm, 2, get_face(perm, 0))  # U = F
        set_face(new_perm, 1, rotate_face_180(get_face(perm, 2)))  # B = U rotado 180
        set_face(new_perm, 3, get_face(perm, 1))  # D = B
        set_face(new_perm, 0, rotate_face_180(get_face(perm, 3)))  # F = D rotado 180
        set_face(new_perm, 5, rotate_face_cw(get_face(perm, 5)))   # R rota CW
        set_face(new_perm, 4, rotate_face_ccw(get_face(perm, 4)))  # L rota CCW
        return new_perm

    # Rotación Y (alrededor del eje U-D): F->L->B->R->F
    def rot_y(perm):
        new_perm = perm.copy()
        set_face(new_perm, 4, get_face(perm, 0))  # L = F
        set_face(new_perm, 1, get_face(perm, 4))  # B = L
        set_face(new_perm, 5, get_face(perm, 1))  # R = B
        set_face(new_perm, 0, get_face(perm, 5))  # F = R
        set_face(new_perm, 2, rotate_face_cw(get_face(perm, 2)))   # U rota CW
        set_face(new_perm, 3, rotate_face_ccw(get_face(perm, 3)))  # D rota CCW
        return new_perm

    # Rotación Z (alrededor del eje F-B): U->R->D->L->U
    def rot_z(perm):
        new_perm = perm.copy()
        set_face(new_perm, 5, rotate_face_cw(get_face(perm, 2)))   # R = U rotado CW
        set_face(new_perm, 3, rotate_face_cw(get_face(perm, 5)))   # D = R rotado CW
        set_face(new_perm, 4, rotate_face_cw(get_face(perm, 3)))   # L = D rotado CW
        set_face(new_perm, 2, rotate_face_cw(get_face(perm, 4)))   # U = L rotado CW
        set_face(new_perm, 0, rotate_face_cw(get_face(perm, 0)))   # F rota CW
        set_face(new_perm, 1, rotate_face_ccw(get_face(perm, 1)))  # B rota CCW
        return new_perm

    # Generar las 24 rotaciones
    rotations = set()
    queue = [identity]
    while queue:
        perm = queue.pop(0)
        perm_tuple = tuple(perm)
        if perm_tuple in rotations:
            continue
        rotations.add(perm_tuple)
        queue.append(rot_x(perm))
        queue.append(rot_y(perm))
        queue.append(rot_z(perm))

    return [list(r) for r in rotations]


# Pre-computar rotaciones como tensor
CUBE_ROTATIONS = create_cube_rotations()  # 24 permutaciones


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

        # Pre-computar índices de permutación para augmentation (24 rotaciones x 324 índices)
        # Cada rotación permuta los 54 stickers, y cada sticker es one-hot de 6 colores
        perm_indices = []
        for rot in CUBE_ROTATIONS:
            indices = []
            for sticker_idx in rot:
                # Cada sticker tiene 6 valores one-hot
                for color in range(6):
                    indices.append(sticker_idx * 6 + color)
            perm_indices.append(indices)
        self.register_buffer('rot_indices', torch.tensor(perm_indices, dtype=torch.long))

    def augment(self, x):
        """Aplica una rotación aleatoria del cubo a cada observación."""
        batch_size = x.shape[0]
        # Seleccionar rotación aleatoria para cada muestra
        rot_idx = torch.randint(0, 24, (batch_size,), device=x.device)
        # Aplicar permutación
        indices = self.rot_indices[rot_idx]  # (batch_size, 324)
        return torch.gather(x, 1, indices)

    def forward(self, x, state=None):
        x = x.float().view(x.shape[0], -1)
        # Augmentation durante training
        if self.training:
            x = self.augment(x)
        x = F.relu(self.input_ln(self.input_fc(x)))
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
    args['train']['total_timesteps'] = 5_000_000_000
    args['train']['learning_rate'] = 3e-4
    args['train']['minibatch_size'] = 65536
    args['train']['bptt_horizon'] = 8
    args['train']['update_epochs'] = 1

    NUM_ENVS = 65536  # Mas envs = mas throughput

    vecenv = pufferlib.vector.make(
        RubiksPufferEnv,
        env_kwargs={'num_envs': NUM_ENVS, 'scramble_moves': 1, 'max_steps': 50},
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )

    device = args['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(vecenv, hidden_size=256, num_blocks=2).to(device)

    print(f"\n  Params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Envs: {NUM_ENVS}")
    print("=" * 60)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    # Curriculum learning settings
    current_scramble = 1
    max_scramble = 20
    success_threshold = 0.7  # 70% solve rate to advance
    check_interval = 50  # Check every N epochs

    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            trainer.train()

            # Curriculum: check solve rate and advance
            if trainer.epoch % check_interval == 0 and current_scramble < max_scramble:
                env = vecenv.envs[0] if hasattr(vecenv, 'envs') else vecenv
                solve_rate = env.get_solve_rate()
                print(f"  [Epoch {trainer.epoch}] Scramble={current_scramble}, Solve rate={solve_rate:.1%}")

                if solve_rate >= success_threshold:
                    current_scramble += 1
                    env.set_scramble(current_scramble)
                    env.reset_stats()

    except KeyboardInterrupt:
        print("\nInterrumpido")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'rubik_puffer.pt')
    print(f"\nFinal scramble level: {current_scramble}")
    trainer.close()
