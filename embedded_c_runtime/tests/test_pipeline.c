#include "runtime.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define W 64
#define H 48

static void fill_scene(uint8_t *under, uint8_t *over) {
    int x, y;
    memset(under, 20, W * H);
    memset(over, 30, W * H);

    for (y = 12; y < 36; ++y) {
        for (x = 18; x < 46; ++x) {
            under[y * W + x] = 120;
            over[y * W + x] = 170;
        }
    }

    for (y = 20; y < 26; ++y) {
        for (x = 30; x < 36; ++x) {
            over[y * W + x] = 255;
        }
    }
}

int main(void) {
    uint8_t under[W * H];
    uint8_t over[W * H];
    uint8_t hdr[W * H];
    uint8_t mask[W * H];
    uint8_t workspace[W * H];
    runtime_ctx_t ctx;
    runtime_config_t cfg;
    runtime_pose_t pose;
    int rc;

    cfg.seg_threshold = 80;
    cfg.specular_clip = 210;
    cfg.specular_blend = 70;
    cfg.width = W;
    cfg.height = H;

    fill_scene(under, over);

    rc = runtime_init(&ctx, &cfg, workspace, sizeof(workspace));
    if (rc != 0) {
        printf("runtime_init failed: %d\n", rc);
        return 1;
    }

    rc = runtime_process_frame(&ctx, under, over, hdr, mask, &pose);
    if (rc != 0) {
        printf("runtime_process_frame failed: %d\n", rc);
        return 2;
    }

    if (pose.foreground_pixels < 500 || pose.foreground_pixels > 900) {
        printf("foreground area out of range: %u\n", pose.foreground_pixels);
        return 3;
    }

    if (fabsf(pose.center_x - 31.5f) > 2.0f || fabsf(pose.center_y - 23.5f) > 2.0f) {
        printf("center mismatch: (%.2f, %.2f)\n", pose.center_x, pose.center_y);
        return 4;
    }

    printf("PASS: area=%u center=(%.2f, %.2f) angle=%.2f\n",
           pose.foreground_pixels,
           pose.center_x,
           pose.center_y,
           pose.angle_deg);
    return 0;
}
