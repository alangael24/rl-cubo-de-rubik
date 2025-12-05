/*
 * rubik2x2.h - High-performance 2x2 Rubik's Cube (Pocket Cube) implementation
 *
 * The 2x2 cube has only corners (no edges or centers).
 * - 6 faces, 4 stickers each = 24 stickers total
 * - 9 possible moves (3 faces × 3 rotations, opposite faces are redundant)
 * - Much easier to solve than 3x3 (God's number = 11 for half-turn metric)
 *
 * Face layout:
 *   0 1
 *   2 3
 *
 * Faces: U(0), D(1), F(2), B(3), L(4), R(5)
 */

#ifndef RUBIK2X2_H
#define RUBIK2X2_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// 2x2 cube constants
#define CUBE2_NUM_FACES 6
#define CUBE2_STICKERS_PER_FACE 4
#define CUBE2_TOTAL_STICKERS 24

// 9 moves: R, R', R2, U, U', U2, F, F', F2
// (L, D, B are redundant due to fixed reference frame)
#define CUBE2_NUM_ACTIONS 9

// Observation size (one-hot: 24 stickers × 6 colors)
#define CUBE2_OBS_SIZE 144

// Move indices
#define MOVE2_R   0
#define MOVE2_RP  1
#define MOVE2_R2  2
#define MOVE2_U   3
#define MOVE2_UP  4
#define MOVE2_U2  5
#define MOVE2_F   6
#define MOVE2_FP  7
#define MOVE2_F2  8

// Face indices
#define FACE2_U 0
#define FACE2_D 1
#define FACE2_F 2
#define FACE2_B 3
#define FACE2_L 4
#define FACE2_R 5

// ============================================================================
// Cube state
// ============================================================================

typedef struct {
    uint8_t state[CUBE2_NUM_FACES][CUBE2_STICKERS_PER_FACE];
} Rubiks2x2;

// ============================================================================
// RNG (xorshift64)
// ============================================================================

static inline uint64_t cube2_xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int cube2_rand_int(uint64_t* state, int max) {
    return (int)(cube2_xorshift64(state) % (uint64_t)max);
}

// ============================================================================
// Cube manipulation
// ============================================================================

static inline void cube2_init(Rubiks2x2* cube) {
    for (int face = 0; face < CUBE2_NUM_FACES; face++) {
        for (int i = 0; i < CUBE2_STICKERS_PER_FACE; i++) {
            cube->state[face][i] = (uint8_t)face;
        }
    }
}

static inline void cube2_copy(Rubiks2x2* dst, const Rubiks2x2* src) {
    memcpy(dst, src, sizeof(Rubiks2x2));
}

static inline bool cube2_is_solved(const Rubiks2x2* cube) {
    for (int face = 0; face < CUBE2_NUM_FACES; face++) {
        uint8_t color = cube->state[face][0];
        for (int i = 1; i < CUBE2_STICKERS_PER_FACE; i++) {
            if (cube->state[face][i] != color) {
                return false;
            }
        }
    }
    return true;
}

static inline int cube2_count_correct(const Rubiks2x2* cube) {
    int count = 0;
    for (int face = 0; face < CUBE2_NUM_FACES; face++) {
        for (int i = 0; i < CUBE2_STICKERS_PER_FACE; i++) {
            if (cube->state[face][i] == face) {
                count++;
            }
        }
    }
    return count;
}

// Rotate face clockwise
// 0 1    2 0
// 2 3 -> 3 1
static inline void cube2_rotate_face_cw(uint8_t face[4]) {
    uint8_t temp = face[0];
    face[0] = face[2];
    face[2] = face[3];
    face[3] = face[1];
    face[1] = temp;
}

// Rotate face counter-clockwise
// 0 1    1 3
// 2 3 -> 0 2
static inline void cube2_rotate_face_ccw(uint8_t face[4]) {
    uint8_t temp = face[0];
    face[0] = face[1];
    face[1] = face[3];
    face[3] = face[2];
    face[2] = temp;
}

// Rotate face 180 degrees
// 0 1    3 2
// 2 3 -> 1 0
static inline void cube2_rotate_face_180(uint8_t face[4]) {
    uint8_t temp0 = face[0];
    uint8_t temp1 = face[1];
    face[0] = face[3];
    face[1] = face[2];
    face[2] = temp1;
    face[3] = temp0;
}

// ============================================================================
// Move implementations
// ============================================================================

// R move (right face clockwise)
static inline void cube2_move_R(Rubiks2x2* cube) {
    cube2_rotate_face_cw(cube->state[FACE2_R]);

    // Save U right column
    uint8_t temp[2] = {cube->state[FACE2_U][1], cube->state[FACE2_U][3]};

    // U[1,3] <- F[1,3]
    cube->state[FACE2_U][1] = cube->state[FACE2_F][1];
    cube->state[FACE2_U][3] = cube->state[FACE2_F][3];

    // F[1,3] <- D[1,3]
    cube->state[FACE2_F][1] = cube->state[FACE2_D][1];
    cube->state[FACE2_F][3] = cube->state[FACE2_D][3];

    // D[1,3] <- B[2,0] (reversed)
    cube->state[FACE2_D][1] = cube->state[FACE2_B][2];
    cube->state[FACE2_D][3] = cube->state[FACE2_B][0];

    // B[0,2] <- temp (reversed)
    cube->state[FACE2_B][0] = temp[1];
    cube->state[FACE2_B][2] = temp[0];
}

// R' move
static inline void cube2_move_RP(Rubiks2x2* cube) {
    cube2_rotate_face_ccw(cube->state[FACE2_R]);

    uint8_t temp[2] = {cube->state[FACE2_U][1], cube->state[FACE2_U][3]};

    // U[1,3] <- B[2,0] (reversed)
    cube->state[FACE2_U][1] = cube->state[FACE2_B][2];
    cube->state[FACE2_U][3] = cube->state[FACE2_B][0];

    // B[0,2] <- D[3,1] (reversed)
    cube->state[FACE2_B][0] = cube->state[FACE2_D][3];
    cube->state[FACE2_B][2] = cube->state[FACE2_D][1];

    // D[1,3] <- F[1,3]
    cube->state[FACE2_D][1] = cube->state[FACE2_F][1];
    cube->state[FACE2_D][3] = cube->state[FACE2_F][3];

    // F[1,3] <- temp
    cube->state[FACE2_F][1] = temp[0];
    cube->state[FACE2_F][3] = temp[1];
}

// R2 move
static inline void cube2_move_R2(Rubiks2x2* cube) {
    cube2_rotate_face_180(cube->state[FACE2_R]);

    // Swap U[1,3] <-> D[1,3]
    uint8_t temp[2] = {cube->state[FACE2_U][1], cube->state[FACE2_U][3]};
    cube->state[FACE2_U][1] = cube->state[FACE2_D][1];
    cube->state[FACE2_U][3] = cube->state[FACE2_D][3];
    cube->state[FACE2_D][1] = temp[0];
    cube->state[FACE2_D][3] = temp[1];

    // Swap F[1,3] <-> B[2,0]
    temp[0] = cube->state[FACE2_F][1];
    temp[1] = cube->state[FACE2_F][3];
    cube->state[FACE2_F][1] = cube->state[FACE2_B][2];
    cube->state[FACE2_F][3] = cube->state[FACE2_B][0];
    cube->state[FACE2_B][2] = temp[0];
    cube->state[FACE2_B][0] = temp[1];
}

// U move (upper face clockwise)
static inline void cube2_move_U(Rubiks2x2* cube) {
    cube2_rotate_face_cw(cube->state[FACE2_U]);

    // Save F top row
    uint8_t temp[2] = {cube->state[FACE2_F][0], cube->state[FACE2_F][1]};

    // F[0,1] <- R[0,1]
    cube->state[FACE2_F][0] = cube->state[FACE2_R][0];
    cube->state[FACE2_F][1] = cube->state[FACE2_R][1];

    // R[0,1] <- B[0,1]
    cube->state[FACE2_R][0] = cube->state[FACE2_B][0];
    cube->state[FACE2_R][1] = cube->state[FACE2_B][1];

    // B[0,1] <- L[0,1]
    cube->state[FACE2_B][0] = cube->state[FACE2_L][0];
    cube->state[FACE2_B][1] = cube->state[FACE2_L][1];

    // L[0,1] <- temp
    cube->state[FACE2_L][0] = temp[0];
    cube->state[FACE2_L][1] = temp[1];
}

// U' move
static inline void cube2_move_UP(Rubiks2x2* cube) {
    cube2_rotate_face_ccw(cube->state[FACE2_U]);

    uint8_t temp[2] = {cube->state[FACE2_F][0], cube->state[FACE2_F][1]};

    // F[0,1] <- L[0,1]
    cube->state[FACE2_F][0] = cube->state[FACE2_L][0];
    cube->state[FACE2_F][1] = cube->state[FACE2_L][1];

    // L[0,1] <- B[0,1]
    cube->state[FACE2_L][0] = cube->state[FACE2_B][0];
    cube->state[FACE2_L][1] = cube->state[FACE2_B][1];

    // B[0,1] <- R[0,1]
    cube->state[FACE2_B][0] = cube->state[FACE2_R][0];
    cube->state[FACE2_B][1] = cube->state[FACE2_R][1];

    // R[0,1] <- temp
    cube->state[FACE2_R][0] = temp[0];
    cube->state[FACE2_R][1] = temp[1];
}

// U2 move
static inline void cube2_move_U2(Rubiks2x2* cube) {
    cube2_rotate_face_180(cube->state[FACE2_U]);

    // Swap F[0,1] <-> B[0,1]
    uint8_t temp[2] = {cube->state[FACE2_F][0], cube->state[FACE2_F][1]};
    cube->state[FACE2_F][0] = cube->state[FACE2_B][0];
    cube->state[FACE2_F][1] = cube->state[FACE2_B][1];
    cube->state[FACE2_B][0] = temp[0];
    cube->state[FACE2_B][1] = temp[1];

    // Swap L[0,1] <-> R[0,1]
    temp[0] = cube->state[FACE2_L][0];
    temp[1] = cube->state[FACE2_L][1];
    cube->state[FACE2_L][0] = cube->state[FACE2_R][0];
    cube->state[FACE2_L][1] = cube->state[FACE2_R][1];
    cube->state[FACE2_R][0] = temp[0];
    cube->state[FACE2_R][1] = temp[1];
}

// F move (front face clockwise)
static inline void cube2_move_F(Rubiks2x2* cube) {
    cube2_rotate_face_cw(cube->state[FACE2_F]);

    // Save U bottom row
    uint8_t temp[2] = {cube->state[FACE2_U][2], cube->state[FACE2_U][3]};

    // U[2,3] <- L[3,1] (rotated)
    cube->state[FACE2_U][2] = cube->state[FACE2_L][3];
    cube->state[FACE2_U][3] = cube->state[FACE2_L][1];

    // L[1,3] <- D[0,1]
    cube->state[FACE2_L][1] = cube->state[FACE2_D][0];
    cube->state[FACE2_L][3] = cube->state[FACE2_D][1];

    // D[0,1] <- R[2,0] (rotated)
    cube->state[FACE2_D][0] = cube->state[FACE2_R][2];
    cube->state[FACE2_D][1] = cube->state[FACE2_R][0];

    // R[0,2] <- temp
    cube->state[FACE2_R][0] = temp[0];
    cube->state[FACE2_R][2] = temp[1];
}

// F' move
static inline void cube2_move_FP(Rubiks2x2* cube) {
    cube2_rotate_face_ccw(cube->state[FACE2_F]);

    uint8_t temp[2] = {cube->state[FACE2_U][2], cube->state[FACE2_U][3]};

    // U[2,3] <- R[0,2]
    cube->state[FACE2_U][2] = cube->state[FACE2_R][0];
    cube->state[FACE2_U][3] = cube->state[FACE2_R][2];

    // R[0,2] <- D[1,0] (rotated)
    cube->state[FACE2_R][0] = cube->state[FACE2_D][1];
    cube->state[FACE2_R][2] = cube->state[FACE2_D][0];

    // D[0,1] <- L[1,3]
    cube->state[FACE2_D][0] = cube->state[FACE2_L][1];
    cube->state[FACE2_D][1] = cube->state[FACE2_L][3];

    // L[1,3] <- temp[1,0] (rotated)
    cube->state[FACE2_L][1] = temp[1];
    cube->state[FACE2_L][3] = temp[0];
}

// F2 move
static inline void cube2_move_F2(Rubiks2x2* cube) {
    cube2_rotate_face_180(cube->state[FACE2_F]);

    // Swap U[2,3] <-> D[1,0]
    uint8_t temp[2] = {cube->state[FACE2_U][2], cube->state[FACE2_U][3]};
    cube->state[FACE2_U][2] = cube->state[FACE2_D][1];
    cube->state[FACE2_U][3] = cube->state[FACE2_D][0];
    cube->state[FACE2_D][0] = temp[1];
    cube->state[FACE2_D][1] = temp[0];

    // Swap L[1,3] <-> R[2,0]
    temp[0] = cube->state[FACE2_L][1];
    temp[1] = cube->state[FACE2_L][3];
    cube->state[FACE2_L][1] = cube->state[FACE2_R][2];
    cube->state[FACE2_L][3] = cube->state[FACE2_R][0];
    cube->state[FACE2_R][0] = temp[1];
    cube->state[FACE2_R][2] = temp[0];
}

// Apply move by action index
static inline void cube2_apply_move(Rubiks2x2* cube, int action) {
    switch (action) {
        case MOVE2_R:  cube2_move_R(cube); break;
        case MOVE2_RP: cube2_move_RP(cube); break;
        case MOVE2_R2: cube2_move_R2(cube); break;
        case MOVE2_U:  cube2_move_U(cube); break;
        case MOVE2_UP: cube2_move_UP(cube); break;
        case MOVE2_U2: cube2_move_U2(cube); break;
        case MOVE2_F:  cube2_move_F(cube); break;
        case MOVE2_FP: cube2_move_FP(cube); break;
        case MOVE2_F2: cube2_move_F2(cube); break;
    }
}

// Scramble
static inline void cube2_scramble(Rubiks2x2* cube, int num_moves, uint64_t* rng) {
    for (int i = 0; i < num_moves; i++) {
        int action = cube2_rand_int(rng, CUBE2_NUM_ACTIONS);
        cube2_apply_move(cube, action);
    }
}

// ============================================================================
// Environment
// ============================================================================

typedef struct {
    Rubiks2x2 cube;

    int scramble_moves;
    int max_steps;
    float solve_reward;
    float step_penalty;
    int reward_mode;

    int step_count;
    int prev_correct;
    bool done;

    uint64_t rng_state;

    float* observations;
    float* rewards;
    uint8_t* terminals;
    uint8_t* truncations;
} Rubik2x2Env;

static inline void env2_init(Rubik2x2Env* env, int scramble_moves, int max_steps,
                             float solve_reward, float step_penalty, int reward_mode,
                             uint64_t seed) {
    cube2_init(&env->cube);
    env->scramble_moves = scramble_moves;
    env->max_steps = max_steps;
    env->solve_reward = solve_reward;
    env->step_penalty = step_penalty;
    env->reward_mode = reward_mode;
    env->step_count = 0;
    env->prev_correct = CUBE2_TOTAL_STICKERS;
    env->done = false;
    env->rng_state = seed ? seed : 42;

    env->observations = NULL;
    env->rewards = NULL;
    env->terminals = NULL;
    env->truncations = NULL;
}

static inline void env2_write_obs(Rubik2x2Env* env) {
    if (env->observations == NULL) return;

    memset(env->observations, 0, CUBE2_OBS_SIZE * sizeof(float));

    for (int face = 0; face < CUBE2_NUM_FACES; face++) {
        for (int i = 0; i < CUBE2_STICKERS_PER_FACE; i++) {
            int sticker_idx = face * CUBE2_STICKERS_PER_FACE + i;
            int color = env->cube.state[face][i];
            int obs_idx = sticker_idx * CUBE2_NUM_FACES + color;
            env->observations[obs_idx] = 1.0f;
        }
    }
}

static inline void env2_reset(Rubik2x2Env* env) {
    cube2_init(&env->cube);
    cube2_scramble(&env->cube, env->scramble_moves, &env->rng_state);
    env->step_count = 0;
    env->prev_correct = cube2_count_correct(&env->cube);
    env->done = false;

    env2_write_obs(env);
}

static inline void env2_step(Rubik2x2Env* env, int action) {
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0;
    if (env->truncations) env->truncations[0] = 0;

    cube2_apply_move(&env->cube, action);
    env->step_count++;

    bool is_solved = cube2_is_solved(&env->cube);
    float reward = 0.0f;

    if (env->reward_mode == 0) {
        // Sparse
        if (is_solved) {
            reward = env->solve_reward;
            if (env->terminals) env->terminals[0] = 1;
        } else {
            reward = -env->step_penalty;
        }
    } else {
        // Dense
        int current_correct = cube2_count_correct(&env->cube);
        float progress = (float)(current_correct - env->prev_correct) / (float)CUBE2_TOTAL_STICKERS;
        env->prev_correct = current_correct;

        if (is_solved) {
            reward = env->solve_reward;
            if (env->terminals) env->terminals[0] = 1;
        } else {
            reward = progress - env->step_penalty;
        }
    }

    if (env->step_count >= env->max_steps && !is_solved) {
        if (env->truncations) env->truncations[0] = 1;
    }

    if (env->rewards) env->rewards[0] = reward;
    env2_write_obs(env);

    if ((env->terminals && env->terminals[0]) || (env->truncations && env->truncations[0])) {
        env2_reset(env);
    }
}

// ============================================================================
// Batch environment
// ============================================================================

typedef struct {
    Rubik2x2Env* envs;
    int num_envs;

    int scramble_moves;
    int max_steps;
    float solve_reward;
    float step_penalty;
    int reward_mode;
} Rubik2x2BatchEnv;

static inline void batch2_env_init(Rubik2x2BatchEnv* batch, int num_envs,
                                   int scramble_moves, int max_steps,
                                   float solve_reward, float step_penalty,
                                   int reward_mode, uint64_t seed) {
    batch->num_envs = num_envs;
    batch->scramble_moves = scramble_moves;
    batch->max_steps = max_steps;
    batch->solve_reward = solve_reward;
    batch->step_penalty = step_penalty;
    batch->reward_mode = reward_mode;

    batch->envs = (Rubik2x2Env*)malloc(num_envs * sizeof(Rubik2x2Env));

    for (int i = 0; i < num_envs; i++) {
        env2_init(&batch->envs[i], scramble_moves, max_steps,
                  solve_reward, step_penalty, reward_mode, seed + i);
    }
}

static inline void batch2_env_free(Rubik2x2BatchEnv* batch) {
    if (batch->envs) {
        free(batch->envs);
        batch->envs = NULL;
    }
}

static inline void batch2_env_set_buffers(Rubik2x2BatchEnv* batch,
                                          float* observations,
                                          float* rewards,
                                          uint8_t* terminals,
                                          uint8_t* truncations) {
    for (int i = 0; i < batch->num_envs; i++) {
        batch->envs[i].observations = observations + i * CUBE2_OBS_SIZE;
        batch->envs[i].rewards = rewards + i;
        batch->envs[i].terminals = terminals + i;
        batch->envs[i].truncations = truncations + i;
    }
}

static inline void batch2_env_reset(Rubik2x2BatchEnv* batch) {
    for (int i = 0; i < batch->num_envs; i++) {
        env2_reset(&batch->envs[i]);
    }
}

static inline void batch2_env_step(Rubik2x2BatchEnv* batch, int* actions) {
    for (int i = 0; i < batch->num_envs; i++) {
        env2_step(&batch->envs[i], actions[i]);
    }
}

static inline void batch2_env_set_scramble(Rubik2x2BatchEnv* batch, int scramble_moves) {
    batch->scramble_moves = scramble_moves;
    for (int i = 0; i < batch->num_envs; i++) {
        batch->envs[i].scramble_moves = scramble_moves;
    }
}

#endif // RUBIK2X2_H
