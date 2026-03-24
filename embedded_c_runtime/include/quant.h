#ifndef EMBEDDED_C_RUNTIME_QUANT_H
#define EMBEDDED_C_RUNTIME_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int8_t quantize_i8(float x, float scale, int32_t zero_point);
float dequantize_i8(int8_t q, float scale, int32_t zero_point);

#ifdef __cplusplus
}
#endif

#endif
