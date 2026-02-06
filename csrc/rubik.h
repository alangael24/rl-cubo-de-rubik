/*
 * rubik.h - High-performance Rubik's Cube 3x3 implementation for PufferLib
 *
 * This implements the Rubik's Cube environment in pure C for maximum
 * performance (targeting 1M+ steps/second).
 *
 * Cube representation:
 * - 6 faces: U(0), D(1), F(2), B(3), L(4), R(5)
 * - Each face has 9 stickers (0-8)
 * - Colors: 0=White, 1=Yellow, 2=Green, 3=Blue, 4=Orange, 5=Red
 *
 * Sticker layout per face:
 *   0 1 2
 *   3 4 5
 *   6 7 8
 */

#ifndef RUBIK_H
#define RUBIK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Face indices
#define FACE_U 0
#define FACE_D 1
#define FACE_F 2
#define FACE_B 3
#define FACE_L 4
#define FACE_R 5

// Number of faces, stickers per face, total stickers
#define NUM_FACES 6
#define STICKERS_PER_FACE 9
#define TOTAL_STICKERS 54

// Number of possible actions (12 moves)
#define NUM_ACTIONS 12

// Action indices
#define MOVE_F  0
#define MOVE_FP 1   // F'
#define MOVE_B  2
#define MOVE_BP 3   // B'
#define MOVE_U  4
#define MOVE_UP 5   // U'
#define MOVE_D  6
#define MOVE_DP 7   // D'
#define MOVE_L  8
#define MOVE_LP 9   // L'
#define MOVE_R  10
#define MOVE_RP 11  // R'

// Observation sizes
#define OBS_TOKEN_SIZE TOTAL_STICKERS
#define OBS_ONEHOT_SIZE (TOTAL_STICKERS * NUM_FACES)
#define OBS_SIZE OBS_ONEHOT_SIZE  // Backward-compatible alias

// Observation encoding modes
#define OBS_MODE_ONEHOT 0
#define OBS_MODE_TOKEN 1

// ============================================================================
// Cube state structure
// ============================================================================

typedef struct {
    uint8_t state[NUM_FACES][STICKERS_PER_FACE];  // Current cube state
    uint8_t solved[NUM_FACES][STICKERS_PER_FACE]; // Solved state for comparison
} RubiksCube;

// ============================================================================
// Environment structure (for PufferLib)
// ============================================================================

typedef struct {
    // Cube state
    RubiksCube cube;

    // Environment parameters
    int scramble_moves;
    int max_steps;
    float solve_reward;
    float step_penalty;
    int reward_mode;  // 0 = sparse, 1 = dense

    // Episode state
    int step_count;
    int prev_correct;
    bool done;

    // RNG state (xorshift64)
    uint64_t rng_state;

    // Observation mode
    int obs_mode;  // OBS_MODE_ONEHOT or OBS_MODE_TOKEN

    // Buffers for PufferLib (pointers to shared memory)
    void* observations;
    float* rewards;
    uint8_t* terminals;
    uint8_t* truncations;
} RubikEnv;

// ============================================================================
// RNG (fast xorshift64)
// ============================================================================

static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int rand_int(uint64_t* state, int max) {
    return (int)(xorshift64(state) % (uint64_t)max);
}

// ============================================================================
// Cube manipulation functions
// ============================================================================

static inline void cube_init(RubiksCube* cube) {
    // Initialize solved state - each face has its own color
    for (int face = 0; face < NUM_FACES; face++) {
        for (int i = 0; i < STICKERS_PER_FACE; i++) {
            cube->state[face][i] = (uint8_t)face;
            cube->solved[face][i] = (uint8_t)face;
        }
    }
}

static inline void cube_reset(RubiksCube* cube) {
    // Reset to solved state
    for (int face = 0; face < NUM_FACES; face++) {
        for (int i = 0; i < STICKERS_PER_FACE; i++) {
            cube->state[face][i] = (uint8_t)face;
        }
    }
}

static inline bool cube_is_solved(const RubiksCube* cube) {
    // Check if cube matches solved state
    for (int face = 0; face < NUM_FACES; face++) {
        for (int i = 0; i < STICKERS_PER_FACE; i++) {
            if (cube->state[face][i] != cube->solved[face][i]) {
                return false;
            }
        }
    }
    return true;
}

static inline int cube_count_correct(const RubiksCube* cube) {
    int count = 0;
    for (int face = 0; face < NUM_FACES; face++) {
        for (int i = 0; i < STICKERS_PER_FACE; i++) {
            if (cube->state[face][i] == cube->solved[face][i]) {
                count++;
            }
        }
    }
    return count;
}

// Rotate a face clockwise (in-place)
static inline void rotate_face_cw(uint8_t face[STICKERS_PER_FACE]) {
    uint8_t temp[STICKERS_PER_FACE];
    memcpy(temp, face, STICKERS_PER_FACE);

    // 0 1 2    6 3 0
    // 3 4 5 -> 7 4 1
    // 6 7 8    8 5 2
    face[0] = temp[6];
    face[1] = temp[3];
    face[2] = temp[0];
    face[3] = temp[7];
    // face[4] stays (center)
    face[5] = temp[1];
    face[6] = temp[8];
    face[7] = temp[5];
    face[8] = temp[2];
}

// Rotate a face counter-clockwise (in-place)
static inline void rotate_face_ccw(uint8_t face[STICKERS_PER_FACE]) {
    uint8_t temp[STICKERS_PER_FACE];
    memcpy(temp, face, STICKERS_PER_FACE);

    // 0 1 2    2 5 8
    // 3 4 5 -> 1 4 7
    // 6 7 8    0 3 6
    face[0] = temp[2];
    face[1] = temp[5];
    face[2] = temp[8];
    face[3] = temp[1];
    // face[4] stays (center)
    face[5] = temp[7];
    face[6] = temp[0];
    face[7] = temp[3];
    face[8] = temp[6];
}

// ============================================================================
// Move implementations
// ============================================================================

static inline void move_F(RubiksCube* cube) {
    rotate_face_cw(cube->state[FACE_F]);

    uint8_t temp[3] = {cube->state[FACE_U][6], cube->state[FACE_U][7], cube->state[FACE_U][8]};

    // U[6,7,8] <- L[8,5,2]
    cube->state[FACE_U][6] = cube->state[FACE_L][8];
    cube->state[FACE_U][7] = cube->state[FACE_L][5];
    cube->state[FACE_U][8] = cube->state[FACE_L][2];

    // L[2,5,8] <- D[0,1,2]
    cube->state[FACE_L][2] = cube->state[FACE_D][0];
    cube->state[FACE_L][5] = cube->state[FACE_D][1];
    cube->state[FACE_L][8] = cube->state[FACE_D][2];

    // D[0,1,2] <- R[6,3,0]
    cube->state[FACE_D][0] = cube->state[FACE_R][6];
    cube->state[FACE_D][1] = cube->state[FACE_R][3];
    cube->state[FACE_D][2] = cube->state[FACE_R][0];

    // R[0,3,6] <- temp
    cube->state[FACE_R][0] = temp[0];
    cube->state[FACE_R][3] = temp[1];
    cube->state[FACE_R][6] = temp[2];
}

static inline void move_F_prime(RubiksCube* cube) {
    rotate_face_ccw(cube->state[FACE_F]);

    uint8_t temp[3] = {cube->state[FACE_U][6], cube->state[FACE_U][7], cube->state[FACE_U][8]};

    // U[6,7,8] <- R[0,3,6]
    cube->state[FACE_U][6] = cube->state[FACE_R][0];
    cube->state[FACE_U][7] = cube->state[FACE_R][3];
    cube->state[FACE_U][8] = cube->state[FACE_R][6];

    // R[0,3,6] <- D[2,1,0]
    cube->state[FACE_R][0] = cube->state[FACE_D][2];
    cube->state[FACE_R][3] = cube->state[FACE_D][1];
    cube->state[FACE_R][6] = cube->state[FACE_D][0];

    // D[0,1,2] <- L[2,5,8]
    cube->state[FACE_D][0] = cube->state[FACE_L][2];
    cube->state[FACE_D][1] = cube->state[FACE_L][5];
    cube->state[FACE_D][2] = cube->state[FACE_L][8];

    // L[2,5,8] <- temp[2,1,0]
    cube->state[FACE_L][2] = temp[2];
    cube->state[FACE_L][5] = temp[1];
    cube->state[FACE_L][8] = temp[0];
}

static inline void move_B(RubiksCube* cube) {
    rotate_face_cw(cube->state[FACE_B]);

    uint8_t temp[3] = {cube->state[FACE_U][0], cube->state[FACE_U][1], cube->state[FACE_U][2]};

    // U[0,1,2] <- R[2,5,8]
    cube->state[FACE_U][0] = cube->state[FACE_R][2];
    cube->state[FACE_U][1] = cube->state[FACE_R][5];
    cube->state[FACE_U][2] = cube->state[FACE_R][8];

    // R[2,5,8] <- D[8,7,6]
    cube->state[FACE_R][2] = cube->state[FACE_D][8];
    cube->state[FACE_R][5] = cube->state[FACE_D][7];
    cube->state[FACE_R][8] = cube->state[FACE_D][6];

    // D[6,7,8] <- L[0,3,6]
    cube->state[FACE_D][6] = cube->state[FACE_L][0];
    cube->state[FACE_D][7] = cube->state[FACE_L][3];
    cube->state[FACE_D][8] = cube->state[FACE_L][6];

    // L[0,3,6] <- temp[2,1,0]
    cube->state[FACE_L][0] = temp[2];
    cube->state[FACE_L][3] = temp[1];
    cube->state[FACE_L][6] = temp[0];
}

static inline void move_B_prime(RubiksCube* cube) {
    rotate_face_ccw(cube->state[FACE_B]);

    uint8_t temp[3] = {cube->state[FACE_U][0], cube->state[FACE_U][1], cube->state[FACE_U][2]};

    // U[0,1,2] <- L[6,3,0]
    cube->state[FACE_U][0] = cube->state[FACE_L][6];
    cube->state[FACE_U][1] = cube->state[FACE_L][3];
    cube->state[FACE_U][2] = cube->state[FACE_L][0];

    // L[0,3,6] <- D[6,7,8]
    cube->state[FACE_L][0] = cube->state[FACE_D][6];
    cube->state[FACE_L][3] = cube->state[FACE_D][7];
    cube->state[FACE_L][6] = cube->state[FACE_D][8];

    // D[6,7,8] <- R[8,5,2]
    cube->state[FACE_D][6] = cube->state[FACE_R][8];
    cube->state[FACE_D][7] = cube->state[FACE_R][5];
    cube->state[FACE_D][8] = cube->state[FACE_R][2];

    // R[2,5,8] <- temp
    cube->state[FACE_R][2] = temp[0];
    cube->state[FACE_R][5] = temp[1];
    cube->state[FACE_R][8] = temp[2];
}

static inline void move_U(RubiksCube* cube) {
    rotate_face_cw(cube->state[FACE_U]);

    uint8_t temp[3] = {cube->state[FACE_F][0], cube->state[FACE_F][1], cube->state[FACE_F][2]};

    // F[0,1,2] <- R[0,1,2]
    cube->state[FACE_F][0] = cube->state[FACE_R][0];
    cube->state[FACE_F][1] = cube->state[FACE_R][1];
    cube->state[FACE_F][2] = cube->state[FACE_R][2];

    // R[0,1,2] <- B[0,1,2]
    cube->state[FACE_R][0] = cube->state[FACE_B][0];
    cube->state[FACE_R][1] = cube->state[FACE_B][1];
    cube->state[FACE_R][2] = cube->state[FACE_B][2];

    // B[0,1,2] <- L[0,1,2]
    cube->state[FACE_B][0] = cube->state[FACE_L][0];
    cube->state[FACE_B][1] = cube->state[FACE_L][1];
    cube->state[FACE_B][2] = cube->state[FACE_L][2];

    // L[0,1,2] <- temp
    cube->state[FACE_L][0] = temp[0];
    cube->state[FACE_L][1] = temp[1];
    cube->state[FACE_L][2] = temp[2];
}

static inline void move_U_prime(RubiksCube* cube) {
    rotate_face_ccw(cube->state[FACE_U]);

    uint8_t temp[3] = {cube->state[FACE_F][0], cube->state[FACE_F][1], cube->state[FACE_F][2]};

    // F[0,1,2] <- L[0,1,2]
    cube->state[FACE_F][0] = cube->state[FACE_L][0];
    cube->state[FACE_F][1] = cube->state[FACE_L][1];
    cube->state[FACE_F][2] = cube->state[FACE_L][2];

    // L[0,1,2] <- B[0,1,2]
    cube->state[FACE_L][0] = cube->state[FACE_B][0];
    cube->state[FACE_L][1] = cube->state[FACE_B][1];
    cube->state[FACE_L][2] = cube->state[FACE_B][2];

    // B[0,1,2] <- R[0,1,2]
    cube->state[FACE_B][0] = cube->state[FACE_R][0];
    cube->state[FACE_B][1] = cube->state[FACE_R][1];
    cube->state[FACE_B][2] = cube->state[FACE_R][2];

    // R[0,1,2] <- temp
    cube->state[FACE_R][0] = temp[0];
    cube->state[FACE_R][1] = temp[1];
    cube->state[FACE_R][2] = temp[2];
}

static inline void move_D(RubiksCube* cube) {
    rotate_face_cw(cube->state[FACE_D]);

    uint8_t temp[3] = {cube->state[FACE_F][6], cube->state[FACE_F][7], cube->state[FACE_F][8]};

    // F[6,7,8] <- L[6,7,8]
    cube->state[FACE_F][6] = cube->state[FACE_L][6];
    cube->state[FACE_F][7] = cube->state[FACE_L][7];
    cube->state[FACE_F][8] = cube->state[FACE_L][8];

    // L[6,7,8] <- B[6,7,8]
    cube->state[FACE_L][6] = cube->state[FACE_B][6];
    cube->state[FACE_L][7] = cube->state[FACE_B][7];
    cube->state[FACE_L][8] = cube->state[FACE_B][8];

    // B[6,7,8] <- R[6,7,8]
    cube->state[FACE_B][6] = cube->state[FACE_R][6];
    cube->state[FACE_B][7] = cube->state[FACE_R][7];
    cube->state[FACE_B][8] = cube->state[FACE_R][8];

    // R[6,7,8] <- temp
    cube->state[FACE_R][6] = temp[0];
    cube->state[FACE_R][7] = temp[1];
    cube->state[FACE_R][8] = temp[2];
}

static inline void move_D_prime(RubiksCube* cube) {
    rotate_face_ccw(cube->state[FACE_D]);

    uint8_t temp[3] = {cube->state[FACE_F][6], cube->state[FACE_F][7], cube->state[FACE_F][8]};

    // F[6,7,8] <- R[6,7,8]
    cube->state[FACE_F][6] = cube->state[FACE_R][6];
    cube->state[FACE_F][7] = cube->state[FACE_R][7];
    cube->state[FACE_F][8] = cube->state[FACE_R][8];

    // R[6,7,8] <- B[6,7,8]
    cube->state[FACE_R][6] = cube->state[FACE_B][6];
    cube->state[FACE_R][7] = cube->state[FACE_B][7];
    cube->state[FACE_R][8] = cube->state[FACE_B][8];

    // B[6,7,8] <- L[6,7,8]
    cube->state[FACE_B][6] = cube->state[FACE_L][6];
    cube->state[FACE_B][7] = cube->state[FACE_L][7];
    cube->state[FACE_B][8] = cube->state[FACE_L][8];

    // L[6,7,8] <- temp
    cube->state[FACE_L][6] = temp[0];
    cube->state[FACE_L][7] = temp[1];
    cube->state[FACE_L][8] = temp[2];
}

static inline void move_L(RubiksCube* cube) {
    rotate_face_cw(cube->state[FACE_L]);

    uint8_t temp[3] = {cube->state[FACE_U][0], cube->state[FACE_U][3], cube->state[FACE_U][6]};

    // U[0,3,6] <- B[8,5,2]
    cube->state[FACE_U][0] = cube->state[FACE_B][8];
    cube->state[FACE_U][3] = cube->state[FACE_B][5];
    cube->state[FACE_U][6] = cube->state[FACE_B][2];

    // B[2,5,8] <- D[6,3,0]
    cube->state[FACE_B][2] = cube->state[FACE_D][6];
    cube->state[FACE_B][5] = cube->state[FACE_D][3];
    cube->state[FACE_B][8] = cube->state[FACE_D][0];

    // D[0,3,6] <- F[0,3,6]
    cube->state[FACE_D][0] = cube->state[FACE_F][0];
    cube->state[FACE_D][3] = cube->state[FACE_F][3];
    cube->state[FACE_D][6] = cube->state[FACE_F][6];

    // F[0,3,6] <- temp
    cube->state[FACE_F][0] = temp[0];
    cube->state[FACE_F][3] = temp[1];
    cube->state[FACE_F][6] = temp[2];
}

static inline void move_L_prime(RubiksCube* cube) {
    rotate_face_ccw(cube->state[FACE_L]);

    uint8_t temp[3] = {cube->state[FACE_U][0], cube->state[FACE_U][3], cube->state[FACE_U][6]};

    // U[0,3,6] <- F[0,3,6]
    cube->state[FACE_U][0] = cube->state[FACE_F][0];
    cube->state[FACE_U][3] = cube->state[FACE_F][3];
    cube->state[FACE_U][6] = cube->state[FACE_F][6];

    // F[0,3,6] <- D[0,3,6]
    cube->state[FACE_F][0] = cube->state[FACE_D][0];
    cube->state[FACE_F][3] = cube->state[FACE_D][3];
    cube->state[FACE_F][6] = cube->state[FACE_D][6];

    // D[0,3,6] <- B[8,5,2]
    cube->state[FACE_D][0] = cube->state[FACE_B][8];
    cube->state[FACE_D][3] = cube->state[FACE_B][5];
    cube->state[FACE_D][6] = cube->state[FACE_B][2];

    // B[2,5,8] <- temp[2,1,0]
    cube->state[FACE_B][2] = temp[2];
    cube->state[FACE_B][5] = temp[1];
    cube->state[FACE_B][8] = temp[0];
}

static inline void move_R(RubiksCube* cube) {
    rotate_face_cw(cube->state[FACE_R]);

    uint8_t temp[3] = {cube->state[FACE_U][2], cube->state[FACE_U][5], cube->state[FACE_U][8]};

    // U[2,5,8] <- F[2,5,8]
    cube->state[FACE_U][2] = cube->state[FACE_F][2];
    cube->state[FACE_U][5] = cube->state[FACE_F][5];
    cube->state[FACE_U][8] = cube->state[FACE_F][8];

    // F[2,5,8] <- D[2,5,8]
    cube->state[FACE_F][2] = cube->state[FACE_D][2];
    cube->state[FACE_F][5] = cube->state[FACE_D][5];
    cube->state[FACE_F][8] = cube->state[FACE_D][8];

    // D[2,5,8] <- B[6,3,0]
    cube->state[FACE_D][2] = cube->state[FACE_B][6];
    cube->state[FACE_D][5] = cube->state[FACE_B][3];
    cube->state[FACE_D][8] = cube->state[FACE_B][0];

    // B[0,3,6] <- temp[2,1,0]
    cube->state[FACE_B][0] = temp[2];
    cube->state[FACE_B][3] = temp[1];
    cube->state[FACE_B][6] = temp[0];
}

static inline void move_R_prime(RubiksCube* cube) {
    rotate_face_ccw(cube->state[FACE_R]);

    uint8_t temp[3] = {cube->state[FACE_U][2], cube->state[FACE_U][5], cube->state[FACE_U][8]};

    // U[2,5,8] <- B[6,3,0]
    cube->state[FACE_U][2] = cube->state[FACE_B][6];
    cube->state[FACE_U][5] = cube->state[FACE_B][3];
    cube->state[FACE_U][8] = cube->state[FACE_B][0];

    // B[0,3,6] <- D[8,5,2]
    cube->state[FACE_B][0] = cube->state[FACE_D][8];
    cube->state[FACE_B][3] = cube->state[FACE_D][5];
    cube->state[FACE_B][6] = cube->state[FACE_D][2];

    // D[2,5,8] <- F[2,5,8]
    cube->state[FACE_D][2] = cube->state[FACE_F][2];
    cube->state[FACE_D][5] = cube->state[FACE_F][5];
    cube->state[FACE_D][8] = cube->state[FACE_F][8];

    // F[2,5,8] <- temp
    cube->state[FACE_F][2] = temp[0];
    cube->state[FACE_F][5] = temp[1];
    cube->state[FACE_F][8] = temp[2];
}

// Apply a move by action index (0-11)
static inline void cube_apply_move(RubiksCube* cube, int action) {
    switch (action) {
        case MOVE_F:  move_F(cube); break;
        case MOVE_FP: move_F_prime(cube); break;
        case MOVE_B:  move_B(cube); break;
        case MOVE_BP: move_B_prime(cube); break;
        case MOVE_U:  move_U(cube); break;
        case MOVE_UP: move_U_prime(cube); break;
        case MOVE_D:  move_D(cube); break;
        case MOVE_DP: move_D_prime(cube); break;
        case MOVE_L:  move_L(cube); break;
        case MOVE_LP: move_L_prime(cube); break;
        case MOVE_R:  move_R(cube); break;
        case MOVE_RP: move_R_prime(cube); break;
    }
}

// Scramble the cube with random moves
static inline void cube_scramble(RubiksCube* cube, int num_moves, uint64_t* rng) {
    for (int i = 0; i < num_moves; i++) {
        int action = rand_int(rng, NUM_ACTIONS);
        cube_apply_move(cube, action);
    }
}

// ============================================================================
// Environment functions
// ============================================================================

static inline void env_init(RubikEnv* env, int scramble_moves, int max_steps,
                           float solve_reward, float step_penalty, int reward_mode,
                           int obs_mode, uint64_t seed) {
    cube_init(&env->cube);
    env->scramble_moves = scramble_moves;
    env->max_steps = max_steps;
    env->solve_reward = solve_reward;
    env->step_penalty = step_penalty;
    env->reward_mode = reward_mode;
    env->step_count = 0;
    env->prev_correct = TOTAL_STICKERS;
    env->done = false;
    env->obs_mode = obs_mode;
    env->rng_state = seed ? seed : 42;

    // Initialize buffer pointers to NULL (will be set by Python)
    env->observations = NULL;
    env->rewards = NULL;
    env->terminals = NULL;
    env->truncations = NULL;
}

// Write observation to buffer in selected encoding
static inline void env_write_obs(RubikEnv* env) {
    if (env->observations == NULL) return;

    if (env->obs_mode == OBS_MODE_TOKEN) {
        uint8_t* out = (uint8_t*)env->observations;
        for (int face = 0; face < NUM_FACES; face++) {
            for (int i = 0; i < STICKERS_PER_FACE; i++) {
                int sticker_idx = face * STICKERS_PER_FACE + i;
                out[sticker_idx] = env->cube.state[face][i];
            }
        }
    } else {
        float* out = (float*)env->observations;
        memset(out, 0, OBS_ONEHOT_SIZE * sizeof(float));
        for (int face = 0; face < NUM_FACES; face++) {
            for (int i = 0; i < STICKERS_PER_FACE; i++) {
                int sticker_idx = face * STICKERS_PER_FACE + i;
                int color = env->cube.state[face][i];
                int obs_idx = sticker_idx * NUM_FACES + color;
                out[obs_idx] = 1.0f;
            }
        }
    }
}

static inline void env_reset(RubikEnv* env) {
    cube_reset(&env->cube);
    cube_scramble(&env->cube, env->scramble_moves, &env->rng_state);
    env->step_count = 0;
    env->prev_correct = cube_count_correct(&env->cube);
    env->done = false;

    // Write observation
    env_write_obs(env);

    // NOTE: Don't clear terminals/truncations/rewards here!
    // They are cleared at the START of env_step(), and we need to
    // preserve them for Python to read after auto-reset.
}

static inline void env_step(RubikEnv* env, int action) {
    // Clear outputs at start
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0;
    if (env->truncations) env->truncations[0] = 0;

    // Apply action
    cube_apply_move(&env->cube, action);
    env->step_count++;

    // Check if solved
    bool is_solved = cube_is_solved(&env->cube);
    float reward = 0.0f;

    if (env->reward_mode == 0) {
        // Sparse reward
        if (is_solved) {
            reward = env->solve_reward;
            if (env->terminals) env->terminals[0] = 1;
        } else {
            reward = -env->step_penalty;
        }
    } else {
        // Dense reward
        int current_correct = cube_count_correct(&env->cube);
        float progress = (float)(current_correct - env->prev_correct) / (float)TOTAL_STICKERS;
        env->prev_correct = current_correct;

        if (is_solved) {
            reward = env->solve_reward;
            if (env->terminals) env->terminals[0] = 1;
        } else {
            reward = progress - env->step_penalty;
        }
    }

    // Check truncation
    if (env->step_count >= env->max_steps && !is_solved) {
        if (env->truncations) env->truncations[0] = 1;
    }

    // Write outputs
    if (env->rewards) env->rewards[0] = reward;
    env_write_obs(env);

    // Auto-reset if done, preserving step outputs for caller.
    if ((env->terminals && env->terminals[0]) || (env->truncations && env->truncations[0])) {
        uint8_t terminal = env->terminals ? env->terminals[0] : 0;
        uint8_t truncation = env->truncations ? env->truncations[0] : 0;
        float step_reward = reward;
        env_reset(env);
        if (env->rewards) env->rewards[0] = step_reward;
        if (env->terminals) env->terminals[0] = terminal;
        if (env->truncations) env->truncations[0] = truncation;
    }
}

// ============================================================================
// Batch environment (for vectorized training)
// ============================================================================

typedef struct {
    RubikEnv* envs;
    int num_envs;

    // Shared parameters
    int scramble_moves;
    int max_steps;
    float solve_reward;
    float step_penalty;
    int reward_mode;
    int obs_mode;
} RubikBatchEnv;

static inline void batch_env_init(RubikBatchEnv* batch, int num_envs,
                                  int scramble_moves, int max_steps,
                                  float solve_reward, float step_penalty,
                                  int reward_mode, int obs_mode, uint64_t seed) {
    batch->num_envs = num_envs;
    batch->scramble_moves = scramble_moves;
    batch->max_steps = max_steps;
    batch->solve_reward = solve_reward;
    batch->step_penalty = step_penalty;
    batch->reward_mode = reward_mode;
    batch->obs_mode = obs_mode;

    batch->envs = (RubikEnv*)malloc(num_envs * sizeof(RubikEnv));

    for (int i = 0; i < num_envs; i++) {
        env_init(&batch->envs[i], scramble_moves, max_steps,
                 solve_reward, step_penalty, reward_mode, obs_mode, seed + i);
    }
}

static inline void batch_env_free(RubikBatchEnv* batch) {
    if (batch->envs) {
        free(batch->envs);
        batch->envs = NULL;
    }
}

static inline void batch_env_set_buffers(RubikBatchEnv* batch,
                                         void* observations,
                                         float* rewards,
                                         uint8_t* terminals,
                                         uint8_t* truncations) {
    for (int i = 0; i < batch->num_envs; i++) {
        if (batch->obs_mode == OBS_MODE_TOKEN) {
            batch->envs[i].observations = ((uint8_t*)observations) + i * OBS_TOKEN_SIZE;
        } else {
            batch->envs[i].observations = ((float*)observations) + i * OBS_ONEHOT_SIZE;
        }
        batch->envs[i].rewards = rewards + i;
        batch->envs[i].terminals = terminals + i;
        batch->envs[i].truncations = truncations + i;
    }
}

static inline void batch_env_reset(RubikBatchEnv* batch) {
    for (int i = 0; i < batch->num_envs; i++) {
        env_reset(&batch->envs[i]);
    }
}

static inline void batch_env_step(RubikBatchEnv* batch, int* actions) {
    for (int i = 0; i < batch->num_envs; i++) {
        env_step(&batch->envs[i], actions[i]);
    }
}

#endif // RUBIK_H
