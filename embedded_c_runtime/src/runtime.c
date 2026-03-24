#include "runtime.h"

#include "ops.h"

#include <math.h>
#include <stddef.h>

static int compute_pose_from_mask(const uint8_t *mask, int width, int height, runtime_pose_t *pose) {
    uint64_t m00 = 0U;
    uint64_t m10 = 0U;
    uint64_t m01 = 0U;
    double mu20 = 0.0;
    double mu02 = 0.0;
    double mu11 = 0.0;
    int x, y;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            uint8_t fg = mask[y * width + x] ? 1U : 0U;
            m00 += fg;
            m10 += (uint64_t)fg * (uint64_t)x;
            m01 += (uint64_t)fg * (uint64_t)y;
        }
    }

    if (m00 == 0U) {
        pose->foreground_pixels = 0U;
        pose->center_x = 0.0f;
        pose->center_y = 0.0f;
        pose->angle_deg = 0.0f;
        return -1;
    }

    {
        double cx = (double)m10 / (double)m00;
        double cy = (double)m01 / (double)m00;

        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
                if (mask[y * width + x]) {
                    double dx = (double)x - cx;
                    double dy = (double)y - cy;
                    mu20 += dx * dx;
                    mu02 += dy * dy;
                    mu11 += dx * dy;
                }
            }
        }

        pose->foreground_pixels = (uint32_t)m00;
        pose->center_x = (float)cx;
        pose->center_y = (float)cy;
        pose->angle_deg = (float)(0.5 * atan2(2.0 * mu11, mu20 - mu02) * 57.29577951308232);
    }

    return 0;
}

int runtime_init(runtime_ctx_t *ctx, const runtime_config_t *cfg, uint8_t *workspace, uint32_t workspace_bytes) {
    uint32_t need;
    if (!ctx || !cfg || !workspace) {
        return -1;
    }
    if (cfg->width <= 2 || cfg->height <= 2) {
        return -2;
    }
    need = (uint32_t)(cfg->width * cfg->height);
    if (workspace_bytes < need) {
        return -3;
    }

    ctx->cfg = *cfg;
    ctx->workspace = workspace;
    ctx->workspace_bytes = workspace_bytes;
    return 0;
}

int runtime_process_frame(runtime_ctx_t *ctx,
                          const uint8_t *under,
                          const uint8_t *over,
                          uint8_t *hdr_out,
                          uint8_t *mask_out,
                          runtime_pose_t *pose) {
    uint32_t pixels;
    if (!ctx || !under || !over || !hdr_out || !mask_out || !pose) {
        return -1;
    }

    pixels = (uint32_t)(ctx->cfg.width * ctx->cfg.height);
    op_hdr_fuse_mean_u8(under, over, hdr_out, pixels);
    op_specular_suppress_u8(hdr_out, pixels, ctx->cfg.specular_clip, ctx->cfg.specular_blend);
    op_threshold_u8(hdr_out, mask_out, ctx->cfg.seg_threshold, pixels);
    op_morph_open3x3_u8(mask_out, mask_out, ctx->workspace, ctx->cfg.width, ctx->cfg.height);

    return compute_pose_from_mask(mask_out, ctx->cfg.width, ctx->cfg.height, pose);
}
