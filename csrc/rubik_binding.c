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
