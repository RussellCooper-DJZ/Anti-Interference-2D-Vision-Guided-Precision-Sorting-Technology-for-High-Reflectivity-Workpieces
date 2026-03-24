#include "quant.h"

static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int8_t quantize_i8(float x, float scale, int32_t zero_point) {
    if (scale <= 0.0f) {
        return 0;
    }
    float inv = x / scale;
    int32_t q = (int32_t)(inv >= 0.0f ? inv + 0.5f : inv - 0.5f) + zero_point;
    q = clamp_i32(q, -128, 127);
    return (int8_t)q;
}

float dequantize_i8(int8_t q, float scale, int32_t zero_point) {
    int32_t centered = (int32_t)q - zero_point;
    return centered * scale;
}
