#ifndef EMBEDDED_C_RUNTIME_TENSOR_H
#define EMBEDDED_C_RUNTIME_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TENSOR_U8 = 0,
    TENSOR_I8 = 1,
    TENSOR_I16 = 2,
    TENSOR_F32 = 3
} tensor_dtype_t;

typedef struct {
    int32_t n;
    int32_t h;
    int32_t w;
    int32_t c;
} tensor_shape_t;

typedef struct {
    float scale;
    int32_t zero_point;
} quant_param_t;

typedef struct {
    tensor_shape_t shape;
    tensor_dtype_t dtype;
    quant_param_t q;
    void *data;
    size_t bytes;
} tensor_t;

size_t tensor_required_bytes(tensor_shape_t shape, tensor_dtype_t dtype);

#ifdef __cplusplus
}
#endif

#endif
