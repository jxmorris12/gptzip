#include "core.h"
#include "bits.h"
#include <string.h>

#define HALF UINT64_C(2147483648)
#define QUARTER UINT64_C(1073741824)
#define MAXIMUM UINT64_C(4294967295)
#define CAPSULE "gptzip.range.v1"

typedef struct {
    uint64_t low, high, code;
    size_t carry;
    int decoding, ended;
    PyObject *input;
    GZWriter writer;
    GZReader reader;
    uint32_t *cdf;
    Py_ssize_t cdf_size;
} Range;

static void destroy(PyObject *capsule) {
    Range *state = PyCapsule_GetPointer(capsule, CAPSULE);
    if (state != NULL) {
        Py_XDECREF(state->input);
        PyMem_RawFree(state->writer.data);
        PyMem_RawFree(state->cdf);
        PyMem_Free(state);
    }
}

static int write_bit(Range *state, unsigned bit) {
    if (gz_write_bit(&state->writer, bit) < 0) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static int read_bit(Range *state, uint64_t *bit) {
    if (gz_read_bit(&state->reader, bit) < 0) {
        PyErr_SetNone(PyExc_StopIteration);
        state->ended = 1;
        return -1;
    }
    return 0;
}

static PyObject *create(PyObject *data, unsigned padding) {
    Range *state = PyMem_Calloc(1, sizeof(Range));
    if (state == NULL) return PyErr_NoMemory();
    state->high = MAXIMUM;
    state->decoding = data != NULL;
    PyObject *capsule = PyCapsule_New(state, CAPSULE, destroy);
    if (capsule == NULL) {
        PyMem_RawFree(state->cdf);
        PyMem_Free(state);
        return NULL;
    }
    if (data != NULL) {
        state->input = Py_NewRef(data);
        state->reader.data = (const unsigned char *)PyBytes_AS_STRING(data);
        state->reader.position = padding;
        state->reader.end = (size_t)PyBytes_GET_SIZE(data) * 8;
        state->reader.trailing = 31;
        for (unsigned i = 0; i < 32; ++i) {
            uint64_t bit;
            if (read_bit(state, &bit) < 0) {
                Py_DECREF(capsule);
                return NULL;
            }
            state->code = state->code * 2 + bit;
        }
    }
    return capsule;
}

PyObject *gz_py_encoder_new(PyObject *module, PyObject *unused) {
    (void)module; (void)unused;
    return create(NULL, 0);
}

PyObject *gz_py_decoder_new(PyObject *module, PyObject *args) {
    (void)module;
    PyObject *data;
    int padding;
    if (!PyArg_ParseTuple(args, "Si", &data, &padding)) return NULL;
    Py_ssize_t size = PyBytes_GET_SIZE(data);
    if (size < 1 || (size_t)size > SIZE_MAX / 8 || padding < 0 || padding > 7) {
        PyErr_SetString(PyExc_ValueError, "Invalid bitstream or padding");
        return NULL;
    }
    return create(data, (unsigned)padding);
}

/* Match NumPy's float64 scaling, including its rounding before conversion.
   An integer product followed by a shift is NOT bit-identical at every bound. */
static uint64_t scaled(uint32_t bound, uint64_t width) {
    double fraction = (double)bound / (double)GZ_TOTAL;
    return (uint64_t)(fraction * (double)width);
}

static int normalize_range(Range *state, uint64_t old_low) {
    while ((state->low >> 31) == (state->high >> 31)) {
        if (!state->decoding) {
            unsigned bit = (unsigned)(state->low >> 31);
            if (write_bit(state, bit) < 0) return -1;
            unsigned carry = (1 + bit - (unsigned)(old_low >> 31)) & 1;
            while (state->carry) {
                if (write_bit(state, carry) < 0) return -1;
                --state->carry;
            }
        } else {
            uint64_t bit;
            if (read_bit(state, &bit) < 0) return -1;
            state->code = (state->code & (HALF - 1)) * 2 + bit;
        }
        state->low = (state->low & (HALF - 1)) * 2;
        state->high = (state->high & (HALF - 1)) * 2 + 1;
    }
    while (state->low / QUARTER + 1 == state->high / QUARTER) {
        if (!state->decoding) {
            ++state->carry;
        } else {
            uint64_t bit;
            if (read_bit(state, &bit) < 0) return -1;
            state->code = (state->code & HALF) + (state->code & (QUARTER - 1)) * 2 + bit;
        }
        state->low = (state->low & HALF) + (state->low & (QUARTER - 1)) * 2;
        state->high = (state->high & HALF) + (state->high & (QUARTER - 1)) * 2 + 1;
    }
    return 0;
}

static int process(Range *state, const uint32_t *cdf, size_t size, Py_ssize_t *symbol) {
    uint64_t width = state->high - state->low + 1;
    if (state->decoding) {
        size_t left = 0, right = size;
        uint64_t position = state->code - state->low;
        while (left < right) {
            size_t middle = left + (right - left) / 2;
            if (scaled(cdf[middle], width) <= position) left = middle + 1;
            else right = middle;
        }
        *symbol = (Py_ssize_t)left - 1;
    }
    if (*symbol < 0 || (size_t)*symbol >= size - 1 ||
        cdf[*symbol] >= cdf[*symbol + 1] || cdf[*symbol + 1] > GZ_TOTAL) {
        PyErr_SetString(PyExc_ValueError, "Invalid CDF interval or symbol");
        return -1;
    }
    uint64_t lower = scaled(cdf[*symbol], width);
    uint64_t upper = scaled(cdf[*symbol + 1], width);
    if (lower >= upper || upper > width) {
        PyErr_SetString(PyExc_ValueError, "Quantized interval collapsed");
        return -1;
    }
    uint64_t old_low = state->low;
    state->low = old_low + lower;
    state->high = old_low + upper - 1;
    return normalize_range(state, old_low);
}

static PyObject *step(PyObject *args, int decoding) {
    PyObject *capsule, *array;
    Py_ssize_t symbol = -1;
    if (decoding) {
        if (!PyArg_ParseTuple(args, "OO", &capsule, &array)) return NULL;
    } else {
        if (!PyArg_ParseTuple(args, "OOn", &capsule, &array, &symbol)) return NULL;
    }
    Range *state = PyCapsule_GetPointer(capsule, CAPSULE);
    if (state == NULL) return NULL;
    if (state->decoding != decoding) {
        PyErr_SetString(PyExc_ValueError, "Encoder/decoder state mismatch");
        return NULL;
    }
    if (state->ended) {
        PyErr_SetString(decoding ? PyExc_StopIteration : PyExc_ValueError, "Coder is terminated");
        return NULL;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(array, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    if (view.ndim != 1 || view.itemsize != sizeof(uint32_t) || view.format == NULL ||
        strcmp(view.format, "I") != 0 || view.len < 2 * (Py_ssize_t)sizeof(uint32_t) ||
        (uintptr_t)view.buf % _Alignof(uint32_t) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Expected aligned contiguous uint32 CDF");
        return NULL;
    }
    const uint32_t *cdf = (const uint32_t *)view.buf;
    size_t size = (size_t)view.len / sizeof(uint32_t);
    if (cdf[0] != 0 || cdf[size - 1] != GZ_TOTAL) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Invalid CDF endpoints");
        return NULL;
    }
    int status = process(state, cdf, size, &symbol);
    PyBuffer_Release(&view);
    if (status < 0) return NULL;
    if (decoding) return PyLong_FromSsize_t(symbol);
    Py_RETURN_NONE;
}

PyObject *gz_py_encode(PyObject *module, PyObject *args) {
    (void)module;
    return step(args, 0);
}

PyObject *gz_py_decode(PyObject *module, PyObject *args) {
    (void)module;
    return step(args, 1);
}

PyObject *gz_py_terminate(PyObject *module, PyObject *capsule) {
    (void)module;
    Range *state = PyCapsule_GetPointer(capsule, CAPSULE);
    if (state == NULL) return NULL;
    if (state->decoding || state->ended) {
        PyErr_SetString(PyExc_ValueError, "Encoder is terminated or state is a decoder");
        return NULL;
    }
    if (write_bit(state, (unsigned)(state->low >> 31)) < 0) return NULL;
    while (state->carry) {
        if (write_bit(state, 1) < 0) return NULL;
        --state->carry;
    }
    state->ended = 1;
    return gz_pack_writer(&state->writer);
}

static PyObject *step_pdf(PyObject *args, int decoding) {
    PyObject *capsule, *array;
    Py_ssize_t symbol = -1;
    if (decoding) {
        if (!PyArg_ParseTuple(args, "OO", &capsule, &array)) return NULL;
    } else {
        if (!PyArg_ParseTuple(args, "OOn", &capsule, &array, &symbol)) return NULL;
    }
    Range *state = PyCapsule_GetPointer(capsule, CAPSULE);
    if (state == NULL) return NULL;
    if (state->decoding != decoding) {
        PyErr_SetString(PyExc_ValueError, "Encoder/decoder state mismatch");
        return NULL;
    }
    if (state->ended) {
        PyErr_SetString(decoding ? PyExc_StopIteration : PyExc_ValueError, "Coder is terminated");
        return NULL;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(array, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return NULL;
    Py_ssize_t size = view.len / sizeof(float);
    if (view.ndim != 1 || view.itemsize != sizeof(float) || view.format == NULL ||
        strcmp(view.format, "f") != 0 || size < 1 || size >= GZ_TOTAL / 2 ||
        (uintptr_t)view.buf % _Alignof(float) != 0 ||
        (!decoding && (symbol < 0 || symbol >= size))) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Invalid float32 probabilities or symbol");
        return NULL;
    }
    if (decoding && state->cdf_size < size + 1) {
        uint32_t *buffer = PyMem_RawRealloc(state->cdf, (size_t)(size + 1) * sizeof(uint32_t));
        if (buffer == NULL) {
            PyBuffer_Release(&view);
            return PyErr_NoMemory();
        }
        state->cdf = buffer;
        state->cdf_size = size + 1;
    }
    uint32_t bounds[2];
    int status;
    /* Retain the GIL: this operation mutates coder-owned scratch space. Input
       buffers must not be mutated by another native thread during the call. */
    if (decoding) status = gz_build_cdf((const float *)view.buf, size, state->cdf);
    else status = gz_encode_bounds((const float *)view.buf, size, symbol, bounds);
    PyBuffer_Release(&view);
    if (status < 0) {
        PyErr_SetString(PyExc_ValueError, "Probabilities must be finite, nonnegative, and nonzero");
        return NULL;
    }
    if (decoding) status = process(state, state->cdf, (size_t)size + 1, &symbol);
    else {
        Py_ssize_t local_symbol = 0;
        status = process(state, bounds, 2, &local_symbol);
    }
    if (status < 0) return NULL;
    if (decoding) return PyLong_FromSsize_t(symbol);
    Py_RETURN_NONE;
}

PyObject *gz_py_encode_pdf(PyObject *module, PyObject *args) {
    (void)module;
    return step_pdf(args, 0);
}

PyObject *gz_py_decode_pdf(PyObject *module, PyObject *args) {
    (void)module;
    return step_pdf(args, 1);
}
