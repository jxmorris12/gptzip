#include "core.h"
#include <math.h>
#include <string.h>

int gz_build_cdf(const float *pdf, Py_ssize_t size, uint32_t *cdf) {
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < size; ++i) {
        double value = (double)pdf[i];
        if (!isfinite(value) || value < 0.0) return -1;
        sum += value;
    }
    if (sum <= 0.0) return -1;
    double prefix = 0.0;
    double budget = (double)(GZ_TOTAL - 2 * (uint32_t)size);
    cdf[0] = 0;
    for (Py_ssize_t i = 0; i < size; ++i) {
        prefix += (double)pdf[i];
        /* Do not replace division with multiplication by a reciprocal or
           contract these operations. Code-word compatibility depends on it. */
        double normalized = prefix / sum;
        double scaled = normalized * budget;
        cdf[i + 1] = (uint32_t)scaled + 2 * (uint32_t)(i + 1);
    }
    return 0;
}

PyObject *gz_py_build_cdf(PyObject *module, PyObject *arg) {
    (void)module;
    Py_buffer view;
    if (PyObject_GetBuffer(arg, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    if (view.ndim != 1 || view.itemsize != sizeof(float) ||
        view.format == NULL || strcmp(view.format, "f") != 0 ||
        (uintptr_t)view.buf % _Alignof(float) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Expected aligned contiguous float32 probabilities");
        return NULL;
    }
    Py_ssize_t size = view.len / sizeof(float);
    if (size < 1 || size >= GZ_TOTAL / 2) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Unsupported vocabulary size");
        return NULL;
    }
    PyObject *result = PyBytes_FromStringAndSize(NULL, (size + 1) * sizeof(uint32_t));
    if (result == NULL) {
        PyBuffer_Release(&view);
        return NULL;
    }
    int status;
    const float *pdf = (const float *)view.buf;
    uint32_t *output = (uint32_t *)PyBytes_AS_STRING(result);
    /* The buffer exports and result reference keep memory alive while unlocked.
       As with NumPy, callers must not mutate input buffers concurrently. */
    Py_BEGIN_ALLOW_THREADS
    status = gz_build_cdf(pdf, size, output);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (status != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Probabilities must be finite, nonnegative, and nonzero");
        return NULL;
    }
    return result;
}

/* Encoding needs only its symbol's two bounds. Preserve the same ordered sum
   and prefixes, but avoid constructing and dividing every other CDF entry. */
int gz_encode_bounds(const float *pdf, Py_ssize_t size, Py_ssize_t symbol, uint32_t *bounds) {
    double sum = 0.0, lower = 0.0, upper = 0.0;
    for (Py_ssize_t i = 0; i < size; ++i) {
        double value = (double)pdf[i];
        if (!isfinite(value) || value < 0.0) return -1;
        if (i == symbol) lower = sum;
        sum += value;
        if (i == symbol) upper = sum;
    }
    if (sum <= 0.0) return -1;
    double budget = (double)(GZ_TOTAL - 2 * (uint32_t)size);
    bounds[0] = (uint32_t)((lower / sum) * budget) + 2 * (uint32_t)symbol;
    bounds[1] = (uint32_t)((upper / sum) * budget) + 2 * (uint32_t)(symbol + 1);
    return 0;
}
