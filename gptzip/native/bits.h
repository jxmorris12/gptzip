#ifndef GPTZIP_BITS_H
#define GPTZIP_BITS_H
#include "core.h"

typedef struct {
    unsigned char *data;
    size_t capacity, count;
} GZWriter;

typedef struct {
    const unsigned char *data;
    size_t position, end;
    unsigned trailing;
} GZReader;

int gz_write_bit(GZWriter *writer, unsigned bit);
int gz_read_bit(GZReader *reader, uint64_t *bit);
PyObject *gz_pack_writer(const GZWriter *writer);
#endif
