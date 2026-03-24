#include "tensor.h"

static size_t dtype_size(tensor_dtype_t dtype) {
    switch (dtype) {
        case TENSOR_U8:
        case TENSOR_I8:
            return 1U;
        case TENSOR_I16:
            return 2U;
        case TENSOR_F32:
            return 4U;
        default:
            return 0U;
    }
}

size_t tensor_required_bytes(tensor_shape_t shape, tensor_dtype_t dtype) {
    if (shape.n <= 0 || shape.h <= 0 || shape.w <= 0 || shape.c <= 0) {
        return 0U;
    }
    return (size_t)shape.n * (size_t)shape.h * (size_t)shape.w * (size_t)shape.c * dtype_size(dtype);
}
