/*
 * rubik_binding.c - Python C extension binding for Rubik's Cube environment
 *
 * This creates a Python module 'rubik_c' that can be imported and used
 * with PufferLib for high-performance training.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "rubik.h"

// ============================================================================
// Python object for single environment
// ============================================================================

typedef struct {
    PyObject_HEAD
    RubikEnv env;
    PyArrayObject* obs_array;
    PyArrayObject* reward_array;
    PyArrayObject* terminal_array;
    PyArrayObject* truncation_array;
} RubikEnvObject;

static void RubikEnvObject_dealloc(RubikEnvObject* self) {
    Py_XDECREF(self->obs_array);
    Py_XDECREF(self->reward_array);
    Py_XDECREF(self->terminal_array);
    Py_XDECREF(self->truncation_array);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* RubikEnvObject_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    RubikEnvObject* self = (RubikEnvObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->obs_array = NULL;
        self->reward_array = NULL;
        self->terminal_array = NULL;
        self->truncation_array = NULL;
    }
    return (PyObject*)self;
}

static int RubikEnvObject_init(RubikEnvObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"scramble_moves", "max_steps", "solve_reward",
                             "step_penalty", "reward_mode", "seed", NULL};

    int scramble_moves = 1;
    int max_steps = 50;
    float solve_reward = 1.0f;
    float step_penalty = 0.01f;
    int reward_mode = 0;  // 0 = sparse
    unsigned long long seed = 42;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiffiK", kwlist,
                                     &scramble_moves, &max_steps, &solve_reward,
                                     &step_penalty, &reward_mode, &seed)) {
        return -1;
    }

    // Initialize environment
    env_init(&self->env, scramble_moves, max_steps, solve_reward,
             step_penalty, reward_mode, (uint64_t)seed);

    // Create numpy arrays for buffers
    npy_intp obs_dims[1] = {OBS_SIZE};
    npy_intp scalar_dims[1] = {1};

    self->obs_array = (PyArrayObject*)PyArray_ZEROS(1, obs_dims, NPY_FLOAT32, 0);
    self->reward_array = (PyArrayObject*)PyArray_ZEROS(1, scalar_dims, NPY_FLOAT32, 0);
    self->terminal_array = (PyArrayObject*)PyArray_ZEROS(1, scalar_dims, NPY_UINT8, 0);
    self->truncation_array = (PyArrayObject*)PyArray_ZEROS(1, scalar_dims, NPY_UINT8, 0);

    if (!self->obs_array || !self->reward_array ||
        !self->terminal_array || !self->truncation_array) {
        return -1;
    }

    // Set buffer pointers
    self->env.observations = (float*)PyArray_DATA(self->obs_array);
    self->env.rewards = (float*)PyArray_DATA(self->reward_array);
    self->env.terminals = (uint8_t*)PyArray_DATA(self->terminal_array);
    self->env.truncations = (uint8_t*)PyArray_DATA(self->truncation_array);

    return 0;
}

static PyObject* RubikEnvObject_reset(RubikEnvObject* self, PyObject* args) {
    PyObject* seed_obj = Py_None;

    if (!PyArg_ParseTuple(args, "|O", &seed_obj)) {
        return NULL;
    }

    if (seed_obj != Py_None) {
        self->env.rng_state = (uint64_t)PyLong_AsUnsignedLongLong(seed_obj);
    }

    env_reset(&self->env);

    // Return observation and empty info dict
    PyObject* info = PyDict_New();
    Py_INCREF(self->obs_array);
    return Py_BuildValue("(OO)", self->obs_array, info);
}

static PyObject* RubikEnvObject_step(RubikEnvObject* self, PyObject* args) {
    int action;

    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }

    if (action < 0 || action >= NUM_ACTIONS) {
        PyErr_SetString(PyExc_ValueError, "Invalid action (must be 0-11)");
        return NULL;
    }

    env_step(&self->env, action);

    // Build info dict
    PyObject* info = PyDict_New();
    PyDict_SetItemString(info, "solved",
        self->env.terminals[0] ? Py_True : Py_False);

    // Return (obs, reward, terminal, truncation, info)
    Py_INCREF(self->obs_array);
    return Py_BuildValue("(OdiiO)",
        self->obs_array,
        (double)self->env.rewards[0],
        (int)self->env.terminals[0],
        (int)self->env.truncations[0],
        info);
}

static PyObject* RubikEnvObject_get_scramble_moves(RubikEnvObject* self, void* closure) {
    return PyLong_FromLong(self->env.scramble_moves);
}

static int RubikEnvObject_set_scramble_moves(RubikEnvObject* self, PyObject* value, void* closure) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "scramble_moves must be an integer");
        return -1;
    }
    int val = (int)PyLong_AsLong(value);
    self->env.scramble_moves = val > 0 ? (val < 26 ? val : 26) : 1;
    return 0;
}

static PyGetSetDef RubikEnvObject_getsetters[] = {
    {"scramble_moves", (getter)RubikEnvObject_get_scramble_moves,
     (setter)RubikEnvObject_set_scramble_moves,
     "Number of scramble moves", NULL},
    {NULL}
};

static PyMethodDef RubikEnvObject_methods[] = {
    {"reset", (PyCFunction)RubikEnvObject_reset, METH_VARARGS,
     "Reset the environment"},
    {"step", (PyCFunction)RubikEnvObject_step, METH_VARARGS,
     "Take a step in the environment"},
    {NULL}
};

static PyTypeObject RubikEnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rubik_c.RubikEnv",
    .tp_doc = "High-performance Rubik's Cube environment",
    .tp_basicsize = sizeof(RubikEnvObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = RubikEnvObject_new,
    .tp_init = (initproc)RubikEnvObject_init,
    .tp_dealloc = (destructor)RubikEnvObject_dealloc,
    .tp_methods = RubikEnvObject_methods,
    .tp_getset = RubikEnvObject_getsetters,
};

// ============================================================================
// Python object for batch/vectorized environment
// ============================================================================

typedef struct {
    PyObject_HEAD
    RubikBatchEnv batch;
    PyArrayObject* obs_array;
    PyArrayObject* reward_array;
    PyArrayObject* terminal_array;
    PyArrayObject* truncation_array;
} RubikBatchEnvObject;

static void RubikBatchEnvObject_dealloc(RubikBatchEnvObject* self) {
    batch_env_free(&self->batch);
    Py_XDECREF(self->obs_array);
    Py_XDECREF(self->reward_array);
    Py_XDECREF(self->terminal_array);
    Py_XDECREF(self->truncation_array);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* RubikBatchEnvObject_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    RubikBatchEnvObject* self = (RubikBatchEnvObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->batch.envs = NULL;
        self->obs_array = NULL;
        self->reward_array = NULL;
        self->terminal_array = NULL;
        self->truncation_array = NULL;
    }
    return (PyObject*)self;
}

static int RubikBatchEnvObject_init(RubikBatchEnvObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"num_envs", "scramble_moves", "max_steps", "solve_reward",
                             "step_penalty", "reward_mode", "seed", NULL};

    int num_envs = 1;
    int scramble_moves = 1;
    int max_steps = 50;
    float solve_reward = 1.0f;
    float step_penalty = 0.01f;
    int reward_mode = 0;
    unsigned long long seed = 42;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiffiK", kwlist,
                                     &num_envs, &scramble_moves, &max_steps,
                                     &solve_reward, &step_penalty, &reward_mode, &seed)) {
        return -1;
    }

    // Initialize batch environment
    batch_env_init(&self->batch, num_envs, scramble_moves, max_steps,
                   solve_reward, step_penalty, reward_mode, (uint64_t)seed);

    // Create numpy arrays
    npy_intp obs_dims[2] = {num_envs, OBS_SIZE};
    npy_intp vec_dims[1] = {num_envs};

    self->obs_array = (PyArrayObject*)PyArray_ZEROS(2, obs_dims, NPY_FLOAT32, 0);
    self->reward_array = (PyArrayObject*)PyArray_ZEROS(1, vec_dims, NPY_FLOAT32, 0);
    self->terminal_array = (PyArrayObject*)PyArray_ZEROS(1, vec_dims, NPY_UINT8, 0);
    self->truncation_array = (PyArrayObject*)PyArray_ZEROS(1, vec_dims, NPY_UINT8, 0);

    if (!self->obs_array || !self->reward_array ||
        !self->terminal_array || !self->truncation_array) {
        return -1;
    }

    // Set buffer pointers
    batch_env_set_buffers(&self->batch,
        (float*)PyArray_DATA(self->obs_array),
        (float*)PyArray_DATA(self->reward_array),
        (uint8_t*)PyArray_DATA(self->terminal_array),
        (uint8_t*)PyArray_DATA(self->truncation_array));

    return 0;
}

static PyObject* RubikBatchEnvObject_reset(RubikBatchEnvObject* self, PyObject* args) {
    batch_env_reset(&self->batch);

    PyObject* info = PyDict_New();
    Py_INCREF(self->obs_array);
    return Py_BuildValue("(OO)", self->obs_array, info);
}

static PyObject* RubikBatchEnvObject_step(RubikBatchEnvObject* self, PyObject* args) {
    PyArrayObject* actions_array;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &actions_array)) {
        return NULL;
    }

    // Ensure actions is the right type and shape
    if (PyArray_NDIM(actions_array) != 1 ||
        PyArray_DIM(actions_array, 0) != self->batch.num_envs) {
        PyErr_SetString(PyExc_ValueError, "actions must be 1D array with num_envs elements");
        return NULL;
    }

    // Convert to int array
    PyArrayObject* actions_int = (PyArrayObject*)PyArray_Cast(actions_array, NPY_INT32);
    if (!actions_int) return NULL;

    int* actions = (int*)PyArray_DATA(actions_int);
    batch_env_step(&self->batch, actions);

    Py_DECREF(actions_int);

    // Return (obs, rewards, terminals, truncations, info)
    PyObject* info = PyDict_New();
    Py_INCREF(self->obs_array);
    Py_INCREF(self->reward_array);
    Py_INCREF(self->terminal_array);
    Py_INCREF(self->truncation_array);

    return Py_BuildValue("(OOOOO)",
        self->obs_array,
        self->reward_array,
        self->terminal_array,
        self->truncation_array,
        info);
}

static PyObject* RubikBatchEnvObject_get_num_envs(RubikBatchEnvObject* self, void* closure) {
    return PyLong_FromLong(self->batch.num_envs);
}

static PyObject* RubikBatchEnvObject_get_scramble_moves(RubikBatchEnvObject* self, void* closure) {
    return PyLong_FromLong(self->batch.scramble_moves);
}

static int RubikBatchEnvObject_set_scramble_moves(RubikBatchEnvObject* self, PyObject* value, void* closure) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "scramble_moves must be an integer");
        return -1;
    }
    int val = (int)PyLong_AsLong(value);
    val = val > 0 ? (val < 26 ? val : 26) : 1;
    self->batch.scramble_moves = val;
    for (int i = 0; i < self->batch.num_envs; i++) {
        self->batch.envs[i].scramble_moves = val;
    }
    return 0;
}

static PyGetSetDef RubikBatchEnvObject_getsetters[] = {
    {"num_envs", (getter)RubikBatchEnvObject_get_num_envs, NULL,
     "Number of environments", NULL},
    {"scramble_moves", (getter)RubikBatchEnvObject_get_scramble_moves,
     (setter)RubikBatchEnvObject_set_scramble_moves,
     "Number of scramble moves", NULL},
    {NULL}
};

static PyMethodDef RubikBatchEnvObject_methods[] = {
    {"reset", (PyCFunction)RubikBatchEnvObject_reset, METH_NOARGS,
     "Reset all environments"},
    {"step", (PyCFunction)RubikBatchEnvObject_step, METH_VARARGS,
     "Take a step in all environments"},
    {NULL}
};

static PyTypeObject RubikBatchEnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rubik_c.RubikBatchEnv",
    .tp_doc = "Vectorized Rubik's Cube environment for batch training",
    .tp_basicsize = sizeof(RubikBatchEnvObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = RubikBatchEnvObject_new,
    .tp_init = (initproc)RubikBatchEnvObject_init,
    .tp_dealloc = (destructor)RubikBatchEnvObject_dealloc,
    .tp_methods = RubikBatchEnvObject_methods,
    .tp_getset = RubikBatchEnvObject_getsetters,
};

// ============================================================================
// PufferLib-native vectorized environment (zero-copy buffers)
// ============================================================================

// Opaque handle for vectorized environments
typedef struct {
    RubikBatchEnv batch;
    int log_interval;
    int tick;
    // Stats for logging
    int* episode_lengths;
    float* episode_returns;
    int* solve_counts;
} VecEnvHandle;

// vec_init: Initialize vectorized environment with external buffers
// This follows the PufferLib Ocean pattern for high performance
static PyObject* rubik_vec_init(PyObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"observations", "actions", "rewards", "terminals",
                             "truncations", "num_envs", "seed",
                             "scramble_moves", "max_steps", "solve_reward",
                             "step_penalty", "reward_mode", "log_interval", NULL};

    PyArrayObject* obs_array;
    PyArrayObject* actions_array;
    PyArrayObject* rewards_array;
    PyArrayObject* terminals_array;
    PyArrayObject* truncations_array;
    int num_envs;
    int seed = 0;
    int scramble_moves = 1;
    int max_steps = 50;
    float solve_reward = 1.0f;
    float step_penalty = 0.01f;
    int reward_mode = 0;
    int log_interval = 128;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!O!ii|iiffii", kwlist,
            &PyArray_Type, &obs_array,
            &PyArray_Type, &actions_array,
            &PyArray_Type, &rewards_array,
            &PyArray_Type, &terminals_array,
            &PyArray_Type, &truncations_array,
            &num_envs, &seed,
            &scramble_moves, &max_steps, &solve_reward,
            &step_penalty, &reward_mode, &log_interval)) {
        return NULL;
    }

    // Allocate handle
    VecEnvHandle* handle = (VecEnvHandle*)malloc(sizeof(VecEnvHandle));
    if (!handle) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate VecEnvHandle");
        return NULL;
    }

    // Initialize batch environment
    batch_env_init(&handle->batch, num_envs, scramble_moves, max_steps,
                   solve_reward, step_penalty, reward_mode, (uint64_t)seed);

    handle->log_interval = log_interval;
    handle->tick = 0;

    // Allocate stats arrays
    handle->episode_lengths = (int*)calloc(num_envs, sizeof(int));
    handle->episode_returns = (float*)calloc(num_envs, sizeof(float));
    handle->solve_counts = (int*)calloc(num_envs, sizeof(int));

    // Set buffer pointers directly from Python arrays (zero-copy!)
    batch_env_set_buffers(&handle->batch,
        (float*)PyArray_DATA(obs_array),
        (float*)PyArray_DATA(rewards_array),
        (uint8_t*)PyArray_DATA(terminals_array),
        (uint8_t*)PyArray_DATA(truncations_array));

    // Store actions pointer in batch for later use
    // (actions are read, not written, so we store separately)
    handle->batch.envs[0].observations = (float*)PyArray_DATA(obs_array);

    return PyLong_FromVoidPtr(handle);
}

// vec_reset: Reset all environments
static PyObject* rubik_vec_reset(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int seed = 0;

    if (!PyArg_ParseTuple(args, "O|i", &handle_obj, &seed)) {
        return NULL;
    }

    VecEnvHandle* handle = (VecEnvHandle*)PyLong_AsVoidPtr(handle_obj);

    // Update seeds if provided
    if (seed != 0) {
        for (int i = 0; i < handle->batch.num_envs; i++) {
            handle->batch.envs[i].rng_state = (uint64_t)(seed + i);
        }
    }

    batch_env_reset(&handle->batch);
    handle->tick = 0;

    // Reset stats
    memset(handle->episode_lengths, 0, handle->batch.num_envs * sizeof(int));
    memset(handle->episode_returns, 0, handle->batch.num_envs * sizeof(float));
    memset(handle->solve_counts, 0, handle->batch.num_envs * sizeof(int));

    Py_RETURN_NONE;
}

// vec_step: Step all environments (reads actions from buffer, writes to obs/rewards/etc)
static PyObject* rubik_vec_step(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    PyArrayObject* actions_array;

    if (!PyArg_ParseTuple(args, "OO!", &handle_obj, &PyArray_Type, &actions_array)) {
        return NULL;
    }

    VecEnvHandle* handle = (VecEnvHandle*)PyLong_AsVoidPtr(handle_obj);
    int* actions = (int*)PyArray_DATA(actions_array);

    // Track stats before step
    for (int i = 0; i < handle->batch.num_envs; i++) {
        handle->episode_lengths[i]++;
    }

    // Step all environments
    batch_env_step(&handle->batch, actions);
    handle->tick++;

    // Track stats after step
    for (int i = 0; i < handle->batch.num_envs; i++) {
        handle->episode_returns[i] += handle->batch.envs[i].rewards[0];

        // If episode ended, track solve and reset stats
        if (handle->batch.envs[i].terminals[0]) {
            handle->solve_counts[i]++;
            handle->episode_lengths[i] = 0;
            handle->episode_returns[i] = 0.0f;
        } else if (handle->batch.envs[i].truncations[0]) {
            handle->episode_lengths[i] = 0;
            handle->episode_returns[i] = 0.0f;
        }
    }

    Py_RETURN_NONE;
}

// vec_log: Return logging info (called periodically)
static PyObject* rubik_vec_log(PyObject* self, PyObject* args) {
    PyObject* handle_obj;

    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    VecEnvHandle* handle = (VecEnvHandle*)PyLong_AsVoidPtr(handle_obj);

    // Calculate aggregate stats
    int total_solves = 0;
    for (int i = 0; i < handle->batch.num_envs; i++) {
        total_solves += handle->solve_counts[i];
    }

    // Build info dict
    PyObject* info = PyDict_New();
    PyDict_SetItemString(info, "tick", PyLong_FromLong(handle->tick));
    PyDict_SetItemString(info, "total_solves", PyLong_FromLong(total_solves));
    PyDict_SetItemString(info, "scramble_moves", PyLong_FromLong(handle->batch.scramble_moves));

    return info;
}

// vec_close: Free resources
static PyObject* rubik_vec_close(PyObject* self, PyObject* args) {
    PyObject* handle_obj;

    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    VecEnvHandle* handle = (VecEnvHandle*)PyLong_AsVoidPtr(handle_obj);

    if (handle) {
        batch_env_free(&handle->batch);
        free(handle->episode_lengths);
        free(handle->episode_returns);
        free(handle->solve_counts);
        free(handle);
    }

    Py_RETURN_NONE;
}

// vec_set_scramble: Update scramble moves dynamically (for curriculum)
static PyObject* rubik_vec_set_scramble(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int scramble_moves;

    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &scramble_moves)) {
        return NULL;
    }

    VecEnvHandle* handle = (VecEnvHandle*)PyLong_AsVoidPtr(handle_obj);

    // Clamp value
    scramble_moves = scramble_moves > 0 ? (scramble_moves < 26 ? scramble_moves : 26) : 1;

    handle->batch.scramble_moves = scramble_moves;
    for (int i = 0; i < handle->batch.num_envs; i++) {
        handle->batch.envs[i].scramble_moves = scramble_moves;
    }

    Py_RETURN_NONE;
}

// vec_render: Render a specific environment (placeholder)
static PyObject* rubik_vec_render(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int env_idx = 0;

    if (!PyArg_ParseTuple(args, "O|i", &handle_obj, &env_idx)) {
        return NULL;
    }

    // Placeholder - could print cube state
    Py_RETURN_NONE;
}

// ============================================================================
// Module definition
// ============================================================================

static PyObject* rubik_c_get_obs_size(PyObject* self, PyObject* args) {
    return PyLong_FromLong(OBS_SIZE);
}

static PyObject* rubik_c_get_num_actions(PyObject* self, PyObject* args) {
    return PyLong_FromLong(NUM_ACTIONS);
}

static PyMethodDef rubik_c_methods[] = {
    {"get_obs_size", rubik_c_get_obs_size, METH_NOARGS,
     "Get observation size (324)"},
    {"get_num_actions", rubik_c_get_num_actions, METH_NOARGS,
     "Get number of actions (12)"},
    // PufferLib-native vectorized functions (zero-copy)
    {"vec_init", (PyCFunction)rubik_vec_init, METH_VARARGS | METH_KEYWORDS,
     "Initialize vectorized environment with external buffers"},
    {"vec_reset", rubik_vec_reset, METH_VARARGS,
     "Reset all environments"},
    {"vec_step", rubik_vec_step, METH_VARARGS,
     "Step all environments"},
    {"vec_log", rubik_vec_log, METH_VARARGS,
     "Get logging info"},
    {"vec_close", rubik_vec_close, METH_VARARGS,
     "Close and free resources"},
    {"vec_set_scramble", rubik_vec_set_scramble, METH_VARARGS,
     "Set scramble moves for curriculum learning"},
    {"vec_render", rubik_vec_render, METH_VARARGS,
     "Render an environment"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rubik_c_module = {
    PyModuleDef_HEAD_INIT,
    "rubik_c",
    "High-performance Rubik's Cube environment in C",
    -1,
    rubik_c_methods
};

PyMODINIT_FUNC PyInit_rubik_c(void) {
    import_array();

    PyObject* m = PyModule_Create(&rubik_c_module);
    if (m == NULL) return NULL;

    if (PyType_Ready(&RubikEnvType) < 0) return NULL;
    if (PyType_Ready(&RubikBatchEnvType) < 0) return NULL;

    Py_INCREF(&RubikEnvType);
    Py_INCREF(&RubikBatchEnvType);

    if (PyModule_AddObject(m, "RubikEnv", (PyObject*)&RubikEnvType) < 0) {
        Py_DECREF(&RubikEnvType);
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "RubikBatchEnv", (PyObject*)&RubikBatchEnvType) < 0) {
        Py_DECREF(&RubikBatchEnvType);
        Py_DECREF(m);
        return NULL;
    }

    // Add constants
    PyModule_AddIntConstant(m, "OBS_SIZE", OBS_SIZE);
    PyModule_AddIntConstant(m, "NUM_ACTIONS", NUM_ACTIONS);

    return m;
}
