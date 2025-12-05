/*
 * rubik2x2_module.c - Python C extension for 2x2 Rubik's Cube
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "rubik2x2.h"

// ============================================================================
// Single Environment
// ============================================================================

typedef struct {
    PyObject_HEAD
    Rubik2x2Env env;
    float obs_buffer[CUBE2_OBS_SIZE];
    float reward_buffer;
    uint8_t terminal_buffer;
    uint8_t truncation_buffer;
} PyRubik2x2Env;

static void PyRubik2x2Env_dealloc(PyRubik2x2Env* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyRubik2x2Env_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyRubik2x2Env* self = (PyRubik2x2Env*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int PyRubik2x2Env_init(PyRubik2x2Env* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"scramble_moves", "max_steps", "solve_reward",
                             "step_penalty", "reward_mode", "seed", NULL};

    int scramble_moves = 1;
    int max_steps = 20;
    float solve_reward = 1.0f;
    float step_penalty = 0.01f;
    int reward_mode = 0;
    unsigned long seed = 42;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiffik", kwlist,
                                      &scramble_moves, &max_steps,
                                      &solve_reward, &step_penalty,
                                      &reward_mode, &seed)) {
        return -1;
    }

    env2_init(&self->env, scramble_moves, max_steps, solve_reward,
              step_penalty, reward_mode, (uint64_t)seed);

    self->env.observations = self->obs_buffer;
    self->env.rewards = &self->reward_buffer;
    self->env.terminals = &self->terminal_buffer;
    self->env.truncations = &self->truncation_buffer;

    return 0;
}

static PyObject* PyRubik2x2Env_reset(PyRubik2x2Env* self, PyObject* args) {
    unsigned long seed = 0;
    if (!PyArg_ParseTuple(args, "|k", &seed)) {
        return NULL;
    }

    if (seed > 0) {
        self->env.rng_state = (uint64_t)seed;
    }

    env2_reset(&self->env);

    npy_intp dims[1] = {CUBE2_OBS_SIZE};
    PyObject* obs = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, self->obs_buffer);

    PyObject* info = PyDict_New();
    return Py_BuildValue("(OO)", obs, info);
}

static PyObject* PyRubik2x2Env_step(PyRubik2x2Env* self, PyObject* args) {
    int action;
    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }

    if (action < 0 || action >= CUBE2_NUM_ACTIONS) {
        PyErr_SetString(PyExc_ValueError, "Invalid action");
        return NULL;
    }

    env2_step(&self->env, action);

    npy_intp dims[1] = {CUBE2_OBS_SIZE};
    PyObject* obs = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, self->obs_buffer);

    PyObject* info = PyDict_New();

    return Py_BuildValue("(OfbbO)",
                         obs,
                         self->reward_buffer,
                         self->terminal_buffer,
                         self->truncation_buffer,
                         info);
}

static PyMethodDef PyRubik2x2Env_methods[] = {
    {"reset", (PyCFunction)PyRubik2x2Env_reset, METH_VARARGS, "Reset environment"},
    {"step", (PyCFunction)PyRubik2x2Env_step, METH_VARARGS, "Step environment"},
    {NULL}
};

static PyTypeObject PyRubik2x2EnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rubik2x2_c.Rubik2x2Env",
    .tp_doc = "2x2 Rubik's Cube Environment",
    .tp_basicsize = sizeof(PyRubik2x2Env),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyRubik2x2Env_new,
    .tp_init = (initproc)PyRubik2x2Env_init,
    .tp_dealloc = (destructor)PyRubik2x2Env_dealloc,
    .tp_methods = PyRubik2x2Env_methods,
};

// ============================================================================
// Batch Environment
// ============================================================================

typedef struct {
    PyObject_HEAD
    Rubik2x2BatchEnv batch;
    PyObject* obs_array;
    PyObject* rewards_array;
    PyObject* terminals_array;
    PyObject* truncations_array;
} PyRubik2x2BatchEnv;

static void PyRubik2x2BatchEnv_dealloc(PyRubik2x2BatchEnv* self) {
    batch2_env_free(&self->batch);
    Py_XDECREF(self->obs_array);
    Py_XDECREF(self->rewards_array);
    Py_XDECREF(self->terminals_array);
    Py_XDECREF(self->truncations_array);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyRubik2x2BatchEnv_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyRubik2x2BatchEnv* self = (PyRubik2x2BatchEnv*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->obs_array = NULL;
        self->rewards_array = NULL;
        self->terminals_array = NULL;
        self->truncations_array = NULL;
    }
    return (PyObject*)self;
}

static int PyRubik2x2BatchEnv_init(PyRubik2x2BatchEnv* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"num_envs", "scramble_moves", "max_steps",
                             "solve_reward", "step_penalty", "reward_mode", "seed", NULL};

    int num_envs = 64;
    int scramble_moves = 1;
    int max_steps = 20;
    float solve_reward = 1.0f;
    float step_penalty = 0.01f;
    int reward_mode = 0;
    unsigned long seed = 42;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiffik", kwlist,
                                      &num_envs, &scramble_moves, &max_steps,
                                      &solve_reward, &step_penalty,
                                      &reward_mode, &seed)) {
        return -1;
    }

    batch2_env_init(&self->batch, num_envs, scramble_moves, max_steps,
                    solve_reward, step_penalty, reward_mode, (uint64_t)seed);

    // Create numpy arrays
    npy_intp obs_dims[2] = {num_envs, CUBE2_OBS_SIZE};
    self->obs_array = PyArray_ZEROS(2, obs_dims, NPY_FLOAT32, 0);

    npy_intp reward_dims[1] = {num_envs};
    self->rewards_array = PyArray_ZEROS(1, reward_dims, NPY_FLOAT32, 0);
    self->terminals_array = PyArray_ZEROS(1, reward_dims, NPY_UINT8, 0);
    self->truncations_array = PyArray_ZEROS(1, reward_dims, NPY_UINT8, 0);

    // Set buffer pointers
    batch2_env_set_buffers(&self->batch,
                           (float*)PyArray_DATA((PyArrayObject*)self->obs_array),
                           (float*)PyArray_DATA((PyArrayObject*)self->rewards_array),
                           (uint8_t*)PyArray_DATA((PyArrayObject*)self->terminals_array),
                           (uint8_t*)PyArray_DATA((PyArrayObject*)self->truncations_array));

    return 0;
}

static PyObject* PyRubik2x2BatchEnv_reset(PyRubik2x2BatchEnv* self, PyObject* args) {
    batch2_env_reset(&self->batch);

    Py_INCREF(self->obs_array);
    PyObject* info = PyDict_New();
    return Py_BuildValue("(OO)", self->obs_array, info);
}

static PyObject* PyRubik2x2BatchEnv_step(PyRubik2x2BatchEnv* self, PyObject* args) {
    PyObject* actions_obj;
    if (!PyArg_ParseTuple(args, "O", &actions_obj)) {
        return NULL;
    }

    PyArrayObject* actions = (PyArrayObject*)PyArray_FROM_OTF(
        actions_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (actions == NULL) {
        return NULL;
    }

    batch2_env_step(&self->batch, (int*)PyArray_DATA(actions));
    Py_DECREF(actions);

    Py_INCREF(self->obs_array);
    Py_INCREF(self->rewards_array);
    Py_INCREF(self->terminals_array);
    Py_INCREF(self->truncations_array);

    PyObject* info = PyDict_New();

    return Py_BuildValue("(OOOOO)",
                         self->obs_array,
                         self->rewards_array,
                         self->terminals_array,
                         self->truncations_array,
                         info);
}

static PyObject* PyRubik2x2BatchEnv_get_scramble_moves(PyRubik2x2BatchEnv* self, void* closure) {
    return PyLong_FromLong(self->batch.scramble_moves);
}

static int PyRubik2x2BatchEnv_set_scramble_moves(PyRubik2x2BatchEnv* self, PyObject* value, void* closure) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "scramble_moves must be an integer");
        return -1;
    }
    int scramble = (int)PyLong_AsLong(value);
    batch2_env_set_scramble(&self->batch, scramble);
    return 0;
}

static PyGetSetDef PyRubik2x2BatchEnv_getsetters[] = {
    {"scramble_moves", (getter)PyRubik2x2BatchEnv_get_scramble_moves,
     (setter)PyRubik2x2BatchEnv_set_scramble_moves, "Number of scramble moves", NULL},
    {NULL}
};

static PyMethodDef PyRubik2x2BatchEnv_methods[] = {
    {"reset", (PyCFunction)PyRubik2x2BatchEnv_reset, METH_VARARGS, "Reset all environments"},
    {"step", (PyCFunction)PyRubik2x2BatchEnv_step, METH_VARARGS, "Step all environments"},
    {NULL}
};

static PyTypeObject PyRubik2x2BatchEnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rubik2x2_c.Rubik2x2BatchEnv",
    .tp_doc = "Batch 2x2 Rubik's Cube Environment",
    .tp_basicsize = sizeof(PyRubik2x2BatchEnv),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyRubik2x2BatchEnv_new,
    .tp_init = (initproc)PyRubik2x2BatchEnv_init,
    .tp_dealloc = (destructor)PyRubik2x2BatchEnv_dealloc,
    .tp_methods = PyRubik2x2BatchEnv_methods,
    .tp_getset = PyRubik2x2BatchEnv_getsetters,
};

// ============================================================================
// Module definition
// ============================================================================

static PyMethodDef module_methods[] = {
    {NULL}
};

static struct PyModuleDef rubik2x2_module = {
    PyModuleDef_HEAD_INIT,
    "rubik2x2_c",
    "High-performance 2x2 Rubik's Cube environment",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_rubik2x2_c(void) {
    import_array();

    PyObject* m = PyModule_Create(&rubik2x2_module);
    if (m == NULL) return NULL;

    if (PyType_Ready(&PyRubik2x2EnvType) < 0) return NULL;
    if (PyType_Ready(&PyRubik2x2BatchEnvType) < 0) return NULL;

    Py_INCREF(&PyRubik2x2EnvType);
    Py_INCREF(&PyRubik2x2BatchEnvType);

    PyModule_AddObject(m, "Rubik2x2Env", (PyObject*)&PyRubik2x2EnvType);
    PyModule_AddObject(m, "Rubik2x2BatchEnv", (PyObject*)&PyRubik2x2BatchEnvType);

    PyModule_AddIntConstant(m, "OBS_SIZE", CUBE2_OBS_SIZE);
    PyModule_AddIntConstant(m, "NUM_ACTIONS", CUBE2_NUM_ACTIONS);
    PyModule_AddIntConstant(m, "TOTAL_STICKERS", CUBE2_TOTAL_STICKERS);

    return m;
}
