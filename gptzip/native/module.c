#include "core.h"

static PyMethodDef methods[] = {
    {"build_cdf", gz_py_build_cdf, METH_O,
     "Construct immutable native-endian uint32 cumulative probabilities."},
    {"encoder_new", gz_py_encoder_new, METH_NOARGS, "Create a binary 32-bit encoder."},
    {"decoder_new", gz_py_decoder_new, METH_VARARGS, "Create a binary 32-bit decoder."},
    {"encode", gz_py_encode, METH_VARARGS, "Encode one symbol under an integer CDF."},
    {"decode", gz_py_decode, METH_VARARGS, "Decode one symbol under an integer CDF."},
    {"encode_pdf", gz_py_encode_pdf, METH_VARARGS, "Encode directly from float32 probabilities."},
    {"decode_pdf", gz_py_decode_pdf, METH_VARARGS, "Decode using reusable CDF storage."},
    {"terminate", gz_py_terminate, METH_O, "Terminate the encoder."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_native",
    .m_doc = "Exact CPU arithmetic coding kernels; fast-math is forbidden.",
    .m_size = -1,
    .m_methods = methods
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
