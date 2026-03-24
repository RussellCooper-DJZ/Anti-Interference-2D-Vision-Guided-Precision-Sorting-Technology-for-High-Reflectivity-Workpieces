#ifndef EMBEDDED_C_RUNTIME_OPS_H
#define EMBEDDED_C_RUNTIME_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void op_hdr_fuse_mean_u8(const uint8_t *under, const uint8_t *over, uint8_t *out, uint32_t count);
void op_specular_suppress_u8(uint8_t *image, uint32_t count, uint8_t clip_threshold, uint8_t blend);
void op_threshold_u8(const uint8_t *src, uint8_t *dst, uint8_t threshold, uint32_t count);
void op_morph_open3x3_u8(const uint8_t *src, uint8_t *dst, uint8_t *tmp, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
