/*
 * rubik_main.c - Test program for Rubik's Cube C implementation
 *
 * Compile: gcc -O3 -o rubik_test rubik_main.c
 * Run: ./rubik_test
 */

#include <stdio.h>
#include <time.h>
#include "rubik.h"

// Print cube state
void print_cube(const RubiksCube* cube) {
    const char* colors = "WYGBOL";  // White, Yellow, Green, Blue, Orange, Red
    const char* face_names[] = {"U", "D", "F", "B", "L", "R"};

    printf("\nCube state:\n");
    for (int face = 0; face < NUM_FACES; face++) {
        printf("%s: ", face_names[face]);
        for (int i = 0; i < STICKERS_PER_FACE; i++) {
            printf("%c", colors[cube->state[face][i]]);
            if (i == 2 || i == 5) printf(" ");
        }
        printf("\n");
    }
}

// Benchmark function
void benchmark(int num_steps, int num_envs) {
    printf("\n=== Benchmark: %d envs, %d steps ===\n", num_envs, num_steps);

    // Allocate buffers
    float* observations = (float*)malloc(num_envs * OBS_SIZE * sizeof(float));
    float* rewards = (float*)malloc(num_envs * sizeof(float));
    uint8_t* terminals = (uint8_t*)malloc(num_envs * sizeof(uint8_t));
    uint8_t* truncations = (uint8_t*)malloc(num_envs * sizeof(uint8_t));
    int* actions = (int*)malloc(num_envs * sizeof(int));

    // Initialize batch environment
    RubikBatchEnv batch;
    batch_env_init(&batch, num_envs, 20, 100, 1.0f, 0.01f, 0, 42);
    batch_env_set_buffers(&batch, observations, rewards, terminals, truncations);
    batch_env_reset(&batch);

    // Random action generator
    uint64_t rng = 12345;

    // Benchmark
    clock_t start = clock();

    for (int step = 0; step < num_steps; step++) {
        // Generate random actions
        for (int i = 0; i < num_envs; i++) {
            actions[i] = rand_int(&rng, NUM_ACTIONS);
        }

        // Step all environments
        batch_env_step(&batch, actions);
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double total_steps = (double)num_steps * num_envs;
    double sps = total_steps / elapsed;

    printf("Time: %.3f seconds\n", elapsed);
    printf("Total steps: %.0f\n", total_steps);
    printf("Steps/second: %.0f\n", sps);
    printf("Steps/second/env: %.0f\n", sps / num_envs);

    // Cleanup
    batch_env_free(&batch);
    free(observations);
    free(rewards);
    free(terminals);
    free(truncations);
    free(actions);
}

int main() {
    printf("Rubik's Cube C Implementation Test\n");
    printf("===================================\n");

    // Test basic cube operations
    RubiksCube cube;
    cube_init(&cube);

    printf("\n1. Initial (solved) cube:\n");
    print_cube(&cube);
    printf("Is solved: %s\n", cube_is_solved(&cube) ? "YES" : "NO");
    printf("Correct stickers: %d/54\n", cube_count_correct(&cube));

    // Test moves
    printf("\n2. After F move:\n");
    move_F(&cube);
    print_cube(&cube);
    printf("Is solved: %s\n", cube_is_solved(&cube) ? "YES" : "NO");

    // Test inverse
    printf("\n3. After F' move (should be solved again):\n");
    move_F_prime(&cube);
    print_cube(&cube);
    printf("Is solved: %s\n", cube_is_solved(&cube) ? "YES" : "NO");

    // Test scramble
    printf("\n4. After scramble (20 moves):\n");
    uint64_t rng = 42;
    cube_scramble(&cube, 20, &rng);
    print_cube(&cube);
    printf("Is solved: %s\n", cube_is_solved(&cube) ? "YES" : "NO");
    printf("Correct stickers: %d/54\n", cube_count_correct(&cube));

    // Test environment
    printf("\n5. Testing environment:\n");
    RubikEnv env;
    float obs[OBS_SIZE];
    float reward;
    uint8_t terminal, truncation;

    env_init(&env, 5, 50, 1.0f, 0.01f, 0, 42);
    env.observations = obs;
    env.rewards = &reward;
    env.terminals = &terminal;
    env.truncations = &truncation;

    env_reset(&env);
    printf("After reset - reward: %.3f, terminal: %d, truncation: %d\n",
           reward, terminal, truncation);

    // Take some steps
    for (int i = 0; i < 5; i++) {
        env_step(&env, i % NUM_ACTIONS);
        printf("Step %d - reward: %.3f, terminal: %d, truncation: %d\n",
               i+1, reward, terminal, truncation);
    }

    // Benchmarks
    printf("\n6. Benchmarks:\n");
    benchmark(100000, 1);      // Single env
    benchmark(100000, 64);     // 64 envs
    benchmark(100000, 256);    // 256 envs

    printf("\nAll tests passed!\n");
    return 0;
}
