#ifndef EMBEDDED_C_RUNTIME_RUNTIME_H
#define EMBEDDED_C_RUNTIME_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t seg_threshold;
    uint8_t specular_clip;
    uint8_t specular_blend;
    int width;
    int height;
} runtime_config_t;

typedef struct {
    float center_x;
    float center_y;
    float angle_deg;
    uint32_t foreground_pixels;
} runtime_pose_t;

typedef struct {
    runtime_config_t cfg;
    uint8_t *workspace;
    uint32_t workspace_bytes;
} runtime_ctx_t;

int runtime_init(runtime_ctx_t *ctx, const runtime_config_t *cfg, uint8_t *workspace, uint32_t workspace_bytes);
int runtime_process_frame(runtime_ctx_t *ctx,
                          const uint8_t *under,
                          const uint8_t *over,
                          uint8_t *hdr_out,
                          uint8_t *mask_out,
                          runtime_pose_t *pose);

#ifdef __cplusplus
}
#endif

#endif
