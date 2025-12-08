"""
symmetry.py - Data augmentation via cube symmetry transformations

The Rubik's Cube has rotational symmetry under the rotation group of the cube,
which has 24 elements. This means if we rotate the entire cube (not individual
faces), the problem remains equivalent but the observation changes.

This module provides:
1. The 24 whole-cube rotation symmetries
2. Functions to apply symmetries to observations
3. Efficient PyTorch implementation for use during training

The observation format is one-hot: 54 stickers × 6 colors = 324 elements.
Sticker ordering: face_idx * 9 + sticker_idx, where faces are U,D,F,B,L,R.
One-hot ordering: sticker_idx * 6 + color_idx.
"""

import numpy as np

# Face indices (matching rubik.h)
FACE_U = 0
FACE_D = 1
FACE_F = 2
FACE_B = 3
FACE_L = 4
FACE_R = 5

NUM_FACES = 6
STICKERS_PER_FACE = 9
TOTAL_STICKERS = 54
OBS_SIZE = 324


def _rotate_stickers_cw(stickers):
    """Rotate 9 sticker values clockwise (as if looking at the face)."""
    # 0 1 2    6 3 0
    # 3 4 5 -> 7 4 1
    # 6 7 8    8 5 2
    return [stickers[6], stickers[3], stickers[0],
            stickers[7], stickers[4], stickers[1],
            stickers[8], stickers[5], stickers[2]]


def _rotate_stickers_ccw(stickers):
    """Rotate 9 sticker values counter-clockwise."""
    # 0 1 2    2 5 8
    # 3 4 5 -> 1 4 7
    # 6 7 8    0 3 6
    return [stickers[2], stickers[5], stickers[8],
            stickers[1], stickers[4], stickers[7],
            stickers[0], stickers[3], stickers[6]]


def _rotate_stickers_180(stickers):
    """Rotate 9 sticker values 180 degrees."""
    # 0 1 2    8 7 6
    # 3 4 5 -> 5 4 3
    # 6 7 8    2 1 0
    return [stickers[8], stickers[7], stickers[6],
            stickers[5], stickers[4], stickers[3],
            stickers[2], stickers[1], stickers[0]]


def _apply_whole_cube_rotation(state, rotation_type):
    """
    Apply a whole-cube rotation to a cube state.

    Args:
        state: list of 54 sticker colors (6 faces × 9 stickers)
        rotation_type: one of 'x', 'x2', 'x3', 'y', 'y2', 'y3', 'z', 'z2', 'z3'

    Returns:
        New state after rotation
    """
    # Convert flat state to 6 faces
    faces = [state[i*9:(i+1)*9] for i in range(6)]
    U, D, F, B, L, R = faces

    if rotation_type == 'x':
        # Rotate around R-L axis: U->F->D->B->U
        # R rotates CW, L rotates CCW
        new_U = _rotate_stickers_180(B)
        new_D = _rotate_stickers_180(F)
        new_F = U[:]
        new_B = D[:]
        new_L = _rotate_stickers_ccw(L)
        new_R = _rotate_stickers_cw(R)
    elif rotation_type == 'x2':
        # x twice
        new_U = D[:]
        new_D = U[:]
        new_F = _rotate_stickers_180(B)
        new_B = _rotate_stickers_180(F)
        new_L = _rotate_stickers_180(L)
        new_R = _rotate_stickers_180(R)
    elif rotation_type == 'x3':
        # x three times = x'
        new_U = F[:]
        new_D = B[:]
        new_F = _rotate_stickers_180(D)
        new_B = _rotate_stickers_180(U)
        new_L = _rotate_stickers_cw(L)
        new_R = _rotate_stickers_ccw(R)
    elif rotation_type == 'y':
        # Rotate around U-D axis: F->L->B->R->F (CW from above)
        # U rotates CW, D rotates CCW
        new_U = _rotate_stickers_cw(U)
        new_D = _rotate_stickers_ccw(D)
        new_F = R[:]
        new_B = L[:]
        new_L = F[:]
        new_R = B[:]
    elif rotation_type == 'y2':
        new_U = _rotate_stickers_180(U)
        new_D = _rotate_stickers_180(D)
        new_F = B[:]
        new_B = F[:]
        new_L = R[:]
        new_R = L[:]
    elif rotation_type == 'y3':
        # y'
        new_U = _rotate_stickers_ccw(U)
        new_D = _rotate_stickers_cw(D)
        new_F = L[:]
        new_B = R[:]
        new_L = B[:]
        new_R = F[:]
    elif rotation_type == 'z':
        # Rotate around F-B axis: U->R->D->L->U (CW from front)
        # F rotates CW, B rotates CCW
        new_U = _rotate_stickers_cw(L)
        new_D = _rotate_stickers_cw(R)
        new_F = _rotate_stickers_cw(F)
        new_B = _rotate_stickers_ccw(B)
        new_L = _rotate_stickers_cw(D)
        new_R = _rotate_stickers_cw(U)
    elif rotation_type == 'z2':
        new_U = _rotate_stickers_180(D)
        new_D = _rotate_stickers_180(U)
        new_F = _rotate_stickers_180(F)
        new_B = _rotate_stickers_180(B)
        new_L = _rotate_stickers_180(R)
        new_R = _rotate_stickers_180(L)
    elif rotation_type == 'z3':
        # z'
        new_U = _rotate_stickers_ccw(R)
        new_D = _rotate_stickers_ccw(L)
        new_F = _rotate_stickers_ccw(F)
        new_B = _rotate_stickers_cw(B)
        new_L = _rotate_stickers_ccw(U)
        new_R = _rotate_stickers_ccw(D)
    else:
        raise ValueError(f"Unknown rotation: {rotation_type}")

    return new_U + new_D + new_F + new_B + new_L + new_R


def _compose_rotations(state, rotations):
    """Apply a sequence of rotations to a state."""
    result = state
    for rot in rotations:
        result = _apply_whole_cube_rotation(result, rot)
    return result


def _generate_all_24_rotations():
    """
    Generate all 24 cube rotations by building permutations.

    We enumerate: 6 faces can be on top × 4 rotations around vertical axis = 24.

    Returns list of sticker permutations, where perm[new_idx] = old_idx.
    """
    # Start with solved cube: sticker i has color i // 9 (face index)
    # For permutation, we track: "where does sticker i go?"
    # or equivalently "what old position contributes to new position j?"

    # Create identity state: position i contains value i
    identity = list(range(TOTAL_STICKERS))

    # All 24 orientations can be reached by:
    # - 6 ways to put a face on top
    # - 4 ways to rotate around vertical axis

    rotation_sequences = [
        # U on top (no x/z needed)
        [],           # identity
        ['y'],        # y
        ['y2'],       # y2
        ['y3'],       # y3

        # D on top (x2)
        ['x2'],
        ['x2', 'y'],
        ['x2', 'y2'],
        ['x2', 'y3'],

        # F on top (x)
        ['x'],
        ['x', 'y'],
        ['x', 'y2'],
        ['x', 'y3'],

        # B on top (x3)
        ['x3'],
        ['x3', 'y'],
        ['x3', 'y2'],
        ['x3', 'y3'],

        # R on top (z3)
        ['z3'],
        ['z3', 'y'],
        ['z3', 'y2'],
        ['z3', 'y3'],

        # L on top (z)
        ['z'],
        ['z', 'y'],
        ['z', 'y2'],
        ['z', 'y3'],
    ]

    permutations = []
    seen = set()

    for seq in rotation_sequences:
        # Apply rotation sequence to identity permutation
        perm = _compose_rotations(identity, seq) if seq else identity[:]

        perm_tuple = tuple(perm)
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            permutations.append(perm)

    assert len(permutations) == 24, f"Expected 24, got {len(permutations)}"

    return permutations


def _sticker_perm_to_obs_perm(sticker_perm):
    """
    Convert a sticker permutation to an observation permutation.

    The observation is one-hot encoded: obs[sticker * 6 + color] = 1 if sticker has that color.

    When we rotate the cube:
    - Sticker at old position i goes to new position j where sticker_perm[j] = i
    - Colors also rotate: if sticker was on face F (color F), after rotation it's on face F'
      where F' is determined by the cube rotation

    For observation permutation: new_obs[j] = old_obs[sticker_perm[j]]
    But since colors are one-hot with 6 values per sticker, we need:
    new_obs[new_sticker * 6 + new_color] = old_obs[old_sticker * 6 + old_color]

    The color mapping is the same as the face mapping: when face F goes to position G,
    color F becomes color G.

    Args:
        sticker_perm: list of 54 ints, where sticker_perm[new_pos] = old_pos

    Returns:
        obs_perm: list of 324 ints for observation permutation
    """
    # First, derive the color/face permutation from the sticker permutation
    # Look at where each center (sticker 4 of each face) goes
    # Center of face F is at position F*9 + 4
    # After rotation, it's at position G*9 + 4 where G is the new face

    # color_perm[new_color] = old_color
    color_perm = [0] * 6
    for new_face in range(6):
        new_center_pos = new_face * 9 + 4
        old_center_pos = sticker_perm[new_center_pos]
        old_face = old_center_pos // 9
        color_perm[new_face] = old_face

    # Now build observation permutation
    # obs_perm[new_idx] = old_idx means: to get new_obs[new_idx], take old_obs[old_idx]
    obs_perm = [0] * OBS_SIZE

    for new_sticker in range(TOTAL_STICKERS):
        old_sticker = sticker_perm[new_sticker]

        for new_color in range(6):
            old_color = color_perm[new_color]

            new_idx = new_sticker * 6 + new_color
            old_idx = old_sticker * 6 + old_color
            obs_perm[new_idx] = old_idx

    return obs_perm


def build_symmetry_permutations():
    """
    Build all 24 symmetry permutations for observations.

    Returns:
        numpy array of shape (24, 324) containing index permutations.
        To apply symmetry i to observation obs:
            new_obs = obs[symmetry_perms[i]]
    """
    sticker_perms = _generate_all_24_rotations()

    obs_perms = []
    for sticker_perm in sticker_perms:
        obs_perm = _sticker_perm_to_obs_perm(sticker_perm)
        obs_perms.append(obs_perm)

    return np.array(obs_perms, dtype=np.int64)


# Pre-compute symmetry permutations at module load
SYMMETRY_PERMUTATIONS = build_symmetry_permutations()
NUM_SYMMETRIES = len(SYMMETRY_PERMUTATIONS)


def apply_random_symmetry_numpy(obs, rng=None):
    """
    Apply a random symmetry to observation(s).

    Args:
        obs: numpy array of shape (..., 324)
        rng: numpy random generator (optional)

    Returns:
        Transformed observation with same shape
    """
    if rng is None:
        rng = np.random.default_rng()

    sym_idx = rng.integers(0, NUM_SYMMETRIES)
    perm = SYMMETRY_PERMUTATIONS[sym_idx]

    return obs[..., perm]


def apply_all_symmetries_numpy(obs):
    """
    Apply all 24 symmetries to an observation.

    Args:
        obs: numpy array of shape (324,) or (batch, 324)

    Returns:
        Array of shape (24, 324) or (batch, 24, 324)
    """
    if obs.ndim == 1:
        return obs[SYMMETRY_PERMUTATIONS]  # (24, 324)
    else:
        # obs is (batch, 324), SYMMETRY_PERMUTATIONS is (24, 324)
        # We want (batch, 24, 324)
        return obs[:, None, :][:, :, SYMMETRY_PERMUTATIONS[0]]  # TODO: fix this


# PyTorch versions for use during training
try:
    import torch

    # Convert to torch tensor (will be moved to device as needed)
    _SYMMETRY_PERMUTATIONS_TORCH = None

    def _get_perms_torch(device):
        global _SYMMETRY_PERMUTATIONS_TORCH
        if _SYMMETRY_PERMUTATIONS_TORCH is None or _SYMMETRY_PERMUTATIONS_TORCH.device != device:
            _SYMMETRY_PERMUTATIONS_TORCH = torch.from_numpy(SYMMETRY_PERMUTATIONS).to(device)
        return _SYMMETRY_PERMUTATIONS_TORCH

    def apply_random_symmetry_batch(obs, generator=None):
        """
        Apply random symmetries to a batch of observations (each gets different symmetry).

        Args:
            obs: torch tensor of shape (batch, 324)
            generator: torch random generator (optional)

        Returns:
            Transformed observations, same shape
        """
        batch_size = obs.shape[0]
        device = obs.device

        perms = _get_perms_torch(device)

        # Random symmetry index for each sample
        if generator is not None:
            sym_indices = torch.randint(0, NUM_SYMMETRIES, (batch_size,),
                                        generator=generator, device=device)
        else:
            sym_indices = torch.randint(0, NUM_SYMMETRIES, (batch_size,), device=device)

        # Gather the permutations for each sample
        selected_perms = perms[sym_indices]  # (batch, 324)

        # Apply permutations using gather
        return torch.gather(obs, 1, selected_perms)

    def apply_symmetry_batch(obs, sym_idx):
        """
        Apply a specific symmetry to a batch of observations.

        Args:
            obs: torch tensor of shape (batch, 324)
            sym_idx: int, symmetry index (0-23)

        Returns:
            Transformed observations
        """
        device = obs.device
        perms = _get_perms_torch(device)
        perm = perms[sym_idx]  # (324,)

        return obs[:, perm]

    def apply_random_symmetry_single(obs):
        """
        Apply a single random symmetry to all observations in a batch.
        More efficient when you want the same transformation for all samples.

        Args:
            obs: torch tensor of shape (batch, 324)

        Returns:
            Transformed observations, same shape, and the symmetry index used
        """
        device = obs.device
        perms = _get_perms_torch(device)

        sym_idx = torch.randint(0, NUM_SYMMETRIES, (1,), device=device).item()
        perm = perms[sym_idx]

        return obs[:, perm], sym_idx

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False


# Verification function
def verify_symmetries():
    """Verify that symmetry permutations are valid."""
    print(f"Generated {NUM_SYMMETRIES} symmetries")

    # Each permutation should be a valid permutation of 0..323
    for i, perm in enumerate(SYMMETRY_PERMUTATIONS):
        assert len(set(perm)) == OBS_SIZE, f"Symmetry {i} is not a valid permutation"
        assert min(perm) == 0 and max(perm) == OBS_SIZE - 1, f"Symmetry {i} has invalid indices"

    # Identity should be first
    assert np.all(SYMMETRY_PERMUTATIONS[0] == np.arange(OBS_SIZE)), "First symmetry should be identity"

    print("All symmetry permutations are valid!")

    # Test with a simple observation
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = 1.0  # First sticker, first color

    for i in range(NUM_SYMMETRIES):
        transformed = obs[SYMMETRY_PERMUTATIONS[i]]
        assert np.sum(transformed) == 1.0, f"Symmetry {i} doesn't preserve one-hot"

    print("One-hot property preserved!")

    # Test that applying symmetry twice with inverse gives identity
    # For rotation group, each element has an inverse

    return True


if __name__ == "__main__":
    verify_symmetries()

    # Benchmark
    import time

    if TORCH_AVAILABLE:
        print("\nBenchmarking PyTorch implementation...")
        obs = torch.randn(256, 324)

        # Warmup
        for _ in range(10):
            _ = apply_random_symmetry_batch(obs)

        start = time.perf_counter()
        for _ in range(1000):
            _ = apply_random_symmetry_batch(obs)
        elapsed = time.perf_counter() - start

        print(f"1000 batches of 256: {elapsed:.3f}s ({elapsed/1000*1e6:.1f} µs per batch)")
