#ifndef GPTZIP_CORE_H
#define GPTZIP_CORE_H
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#define GZ_TOTAL UINT32_C(1073741824)

/* Keep arithmetic order identical to the frozen float64 NumPy profile. */
PyObject *gz_py_build_cdf(PyObject *module, PyObject *arg);

PyObject *gz_py_encoder_new(PyObject *module, PyObject *arg);
PyObject *gz_py_decoder_new(PyObject *module, PyObject *arg);
PyObject *gz_py_encode(PyObject *module, PyObject *args);
PyObject *gz_py_decode(PyObject *module, PyObject *args);
PyObject *gz_py_terminate(PyObject *module, PyObject *arg);

int gz_build_cdf(const float *pdf, Py_ssize_t size, uint32_t *cdf);
int gz_encode_bounds(const float *pdf, Py_ssize_t size, Py_ssize_t symbol, uint32_t *bounds);
PyObject *gz_py_encode_pdf(PyObject *module, PyObject *args);
PyObject *gz_py_decode_pdf(PyObject *module, PyObject *args);

#endif
