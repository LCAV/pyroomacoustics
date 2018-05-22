
#include <stdio.h>
#include <string.h>
#include "room.h"
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject*
py_check_visibility_all(PyObject *self, PyObject *args)
{
  // Argument 1 (expecting ctypes.Structure)
  Py_buffer arg1;
  room_t *room;

  /*
   * w*: the room c structure and its size
   */

  if (!PyArg_ParseTuple(args, "w*", &arg1))
        return NULL;

  // Check the size of the first argument
  if (arg1.len != sizeof(room_t)) {
      PyErr_SetString(PyExc_TypeError, "wrong buffer size");
      return NULL;
  }

  room = arg1.buf;

  // free resources allocated here
  check_visibility_all(room);

  PyBuffer_Release(&arg1);

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject*
py_image_source_model(PyObject *dummy, PyObject *args)
{
  // Argument 1 (expecting ctypes.Structure)
  Py_buffer arg1;
  room_t *room = NULL;

  // Argument 2 (expecting read-only numpy array, np.float32)
  PyObject *arg2 = NULL, *arr2 = NULL;
  int arr2_nd;
  npy_intp *arr2_shape;
  float *source_location = NULL;

  // Argument 3 (expecting integer)
  int max_order;

  // other variables
  int ret;

  /*
   * w#: the room c structure and its size
   * O : a numpy array
   * i : an integer
   */

  if (!PyArg_ParseTuple(args, "w*Oi", &arg1, &arg2, &max_order))
        return NULL;

  // Check the size of the first argument
  if (arg1.len != sizeof(room_t)) {
      PyErr_SetString(PyExc_TypeError, "wrong buffer size");
      return NULL;
  }
  room = (room_t *)arg1.buf;

  // Check second argument
  arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT32, NPY_IN_ARRAY);

  if (arr2 == NULL) 
  {
    PyErr_SetString(PyExc_TypeError, "Could not get pointer to source location");
    goto fail;
  }

  arr2_nd = PyArray_NDIM(arr2);   // check 2 dimensional
  if (arr2_nd != 1)
  {
    PyErr_SetString(PyExc_TypeError, "Source location should be one dimensional");
    goto fail;
  }

  arr2_shape = PyArray_DIMS(arr2);  // npy_intp array of length nd showing length in each dim.
  if (arr2_shape[0] != room->dim) 
  {
    PyErr_SetString(PyExc_TypeError, "Source location as many elements as there are dimensions ");
    goto fail;
  }

  // get arg2 data
  source_location = (float *)PyArray_DATA(arr2);

  /* do function */
  ret = image_source_model(room, source_location, max_order);
  if (ret < 0)
  {
    PyErr_SetString(PyExc_TypeError, "Memory could not be allocated.");
    goto fail;
  }

  PyBuffer_Release(&arg1);

  /* return something */
  Py_DECREF(arr2);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  PyBuffer_Release(&arg1);
  Py_XDECREF(arr2);
  return NULL;
}

static PyObject*
py_image_source_shoebox(PyObject *self, PyObject *args)
{
  // Argument 1 (expecting ctypes.Structure)
  Py_buffer arg1;
  room_t *room = NULL;

  // Use for all nd-array arguments
  int nd;
  npy_intp *shape = NULL;

  // Argument 2 (expecting read-only numpy array, np.float32)
  PyObject *arg2 = NULL;
  PyArrayObject *arr2 = NULL;
  float *source_location = NULL;

  // Argument 3 (expecting read-only numpy array, np.float32)
  PyObject *arg3 = NULL;
  PyArrayObject *arr3 = NULL;
  float *room_size = NULL;

  // Argument 4 (expecting read-only numpy array, np.float32)
  PyObject *arg4 = NULL;
  PyArrayObject *arr4 = NULL;
  float *absorption = NULL;

  // Argument 3 (expecting integer)
  int max_order;

  // other variables
  int ret;

  /* convert Python arguments */

  /*
   * w#: the room c structure and its size, size == sizeof(room_t)
   * O : source location, a numpy float array, length == room->dim
   * O : room size, a numpy float array, length == room->dim
   * O : absorption list, a numpy float array, length == 2 * room->dim
   * i : maximum order, an integer
   */

  if (!PyArg_ParseTuple(args, "w*OOOi", &arg1, &arg2, &arg3, &arg4, &max_order))
        return NULL;

  // Check the size of the first argument
  if (arg1.len != sizeof(room_t)) {
      PyErr_SetString(PyExc_TypeError, "wrong buffer size");
      return NULL;
  }

  room = arg1.buf;

  // Check second argument
  arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr2== NULL) 
  {
    PyErr_SetString(PyExc_TypeError, "Could not get pointer to source location");
    goto fail;
  }

  nd = PyArray_NDIM(arr2);   // check 2 dimensional
  if (nd != 1)
  {
    PyErr_SetString(PyExc_TypeError, "Source location should be one dimensional");
    goto fail;
  }

  shape = PyArray_DIMS(arr2);  // npy_intp array of length nd showing length in each dim.
  if (shape[0] != room->dim) 
  {
    PyErr_SetString(PyExc_TypeError, "Source location as many elements as there are dimensions ");
    goto fail;
  }

  // get arg2 data
  source_location = (float *)PyArray_DATA(arr2);

  // Check third argument
  arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr3 == NULL) 
  {
    PyErr_SetString(PyExc_TypeError, "Could not get pointer to room size");
    goto fail;
  }

  nd = PyArray_NDIM(arr3);   // check 1 dimensional
  if (nd != 1)
  {
    PyErr_SetString(PyExc_TypeError, "Room size should be one dimensional");
    goto fail;
  }

  shape = PyArray_DIMS(arr3);  // npy_intp array of length nd showing length in each dim.
  if (shape[0] != room->dim) 
  {
    PyErr_SetString(PyExc_TypeError, "Room size as many elements as there are dimensions ");
    goto fail;
  }

  // get arg3 data
  room_size = (float *)PyArray_DATA(arr3);

  // Check fourth argument
  arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_FLOAT, NPY_IN_ARRAY);
  if (arr4 == NULL) 
  {
    PyErr_SetString(PyExc_TypeError, "Could not get pointer to absorption");
    goto fail;
  }

  nd = PyArray_NDIM(arr4);   // check 1 dimensional
  if (nd != 1)
  {
    PyErr_SetString(PyExc_TypeError, "Absorption array should be one dimensional");
    goto fail;
  }

  shape = PyArray_DIMS(arr4);  // npy_intp array of length nd showing length in each dim.
  if (shape[0] != 2 * room->dim) 
  {
    PyErr_SetString(PyExc_TypeError, "There should as many absorption coefficients as walls");
    goto fail;
  }

  // get arg3 data
  absorption = (float *)PyArray_DATA(arr4);

  /* do function */
  ret = image_source_shoebox(room, source_location, room_size, absorption, max_order);
  if (ret < 0)
  {
    PyErr_SetString(PyExc_TypeError, "Memory could not be allocated.");
    goto fail;
  }

  PyBuffer_Release(&arg1);
  Py_DECREF(arr2);
  Py_DECREF(arr3);
  Py_DECREF(arr4);

  Py_INCREF(Py_None);
  return Py_None;

fail:
  PyBuffer_Release(&arg1);
  Py_XDECREF(arr2);
  Py_XDECREF(arr3);
  Py_XDECREF(arr4);

  return NULL;
}

static PyObject*
py_free_sources(PyObject *self, PyObject *args)
{
  // Argument 1 (expecting ctypes.Structure)
  Py_buffer arg1;
  room_t *room;

  /*
   * t#: the room c structure and its size
   */

  if (!PyArg_ParseTuple(args, "w*", &arg1))
        return NULL;

  // Check the size of the first argument
  if (arg1.len != sizeof(room_t)) {
      PyErr_SetString(PyExc_TypeError, "wrong buffer size");
      return NULL;
  }

  room = arg1.buf;

  // free resources allocated here
  free_sources(room);

  PyBuffer_Release(&arg1);

  Py_INCREF(Py_None);
  return Py_None;
}


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef libroom_methods[] = {
    {"check_visibility_all", (PyCFunction)py_check_visibility_all, METH_VARARGS, "Check visibility of all sources."},
    {"image_source_model", (PyCFunction)py_image_source_model, METH_VARARGS, "Compute image source model on arbitrary polyhedral room."},
    {"image_source_shoebox", (PyCFunction)py_image_source_shoebox, METH_VARARGS, "Compute image source model on arbitrary polyhedral room."},
    {"free_sources", (PyCFunction)py_free_sources, METH_VARARGS, "Free the resources used to store the image sources."},
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static int libroom_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int libroom_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "libroom",
        NULL,
        sizeof(struct module_state),
        libroom_methods,
        NULL,
        libroom_traverse,
        libroom_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_libroom(void)

#else
#define INITERROR return

void
initlibroom(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("libroom", libroom_methods);
#endif

    import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("libroom.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

