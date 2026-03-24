#include "ops.h"

#include <string.h>

void op_hdr_fuse_mean_u8(const uint8_t *under, const uint8_t *over, uint8_t *out, uint32_t count) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
        out[i] = (uint8_t)((((uint16_t)under[i]) + ((uint16_t)over[i])) >> 1);
    }
}

void op_specular_suppress_u8(uint8_t *image, uint32_t count, uint8_t clip_threshold, uint8_t blend) {
    uint8_t clamped_blend = blend > 100U ? 100U : blend;
    uint32_t i;
    for (i = 0; i < count; ++i) {
        if (image[i] > clip_threshold) {
            uint16_t clipped = (uint16_t)clip_threshold;
            uint16_t original = (uint16_t)image[i];
            uint16_t mixed = (uint16_t)((clamped_blend * clipped + (100U - clamped_blend) * original) / 100U);
            image[i] = (uint8_t)mixed;
        }
    }
}

void op_threshold_u8(const uint8_t *src, uint8_t *dst, uint8_t threshold, uint32_t count) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
        dst[i] = src[i] >= threshold ? 255U : 0U;
    }
}

static uint8_t erode3x3_at(const uint8_t *src, int x, int y, int width) {
    int dx, dy;
    for (dy = -1; dy <= 1; ++dy) {
        for (dx = -1; dx <= 1; ++dx) {
            if (src[(y + dy) * width + (x + dx)] == 0U) {
                return 0U;
            }
        }
    }
    return 255U;
}

static uint8_t dilate3x3_at(const uint8_t *src, int x, int y, int width) {
    int dx, dy;
    for (dy = -1; dy <= 1; ++dy) {
        for (dx = -1; dx <= 1; ++dx) {
            if (src[(y + dy) * width + (x + dx)] != 0U) {
                return 255U;
            }
        }
    }
    return 0U;
}

void op_morph_open3x3_u8(const uint8_t *src, uint8_t *dst, uint8_t *tmp, int width, int height) {
    int x, y;

    memset(tmp, 0, (size_t)width * (size_t)height);
    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            tmp[y * width + x] = erode3x3_at(src, x, y, width);
        }
    }

    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            dst[y * width + x] = dilate3x3_at(tmp, x, y, width);
        }
    }
}
