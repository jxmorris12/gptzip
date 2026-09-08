#include "bits.h"

int gz_write_bit(GZWriter *writer, unsigned bit) {
    if (writer->count / 8 == writer->capacity) {
        if (writer->capacity > SIZE_MAX / 16) return -1;
        size_t capacity = writer->capacity ? writer->capacity * 2 : 128;
        unsigned char *data = PyMem_RawRealloc(writer->data, capacity);
        if (data == NULL) return -1;
        writer->data = data;
        writer->capacity = capacity;
    }
    size_t index = writer->count / 8;
    unsigned offset = (unsigned)(writer->count % 8);
    if (offset == 0) writer->data[index] = 0;
    writer->data[index] |= (unsigned char)(bit << (7 - offset));
    ++writer->count;
    return 0;
}

int gz_read_bit(GZReader *reader, uint64_t *bit) {
    if (reader->position < reader->end) {
        size_t index = reader->position / 8;
        unsigned shift = 7 - (unsigned)(reader->position % 8);
        *bit = (reader->data[index] >> shift) & 1;
        ++reader->position;
        return 0;
    }
    if (reader->trailing) {
        --reader->trailing;
        *bit = 1;
        return 0;
    }
    return -1;
}

PyObject *gz_pack_writer(const GZWriter *writer) {
    size_t length = (writer->count + 7) / 8;
    if (length > PY_SSIZE_T_MAX) return PyErr_NoMemory();
    unsigned padding = (unsigned)((8 - writer->count % 8) % 8);
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)length);
    if (result == NULL) return NULL;
    unsigned char *output = (unsigned char *)PyBytes_AS_STRING(result);
    unsigned carry = 0;
    /* The pilot pads on the LEFT, not the right. Shift the packed word to
       preserve the exact legacy representation without expanding it to bits. */
    for (size_t i = 0; i < length; ++i) {
        unsigned value = writer->data[i];
        output[i] = (unsigned char)(carry | (value >> padding));
        carry = (value << (8 - padding)) & 255;
    }
    return Py_BuildValue("NI", result, padding);
}
