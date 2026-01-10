#include <stdint.h>
#include "arm_helium_utils.h" // 假设包含 Helium 相关的宏和内联函数

/**
 * 使用 Helium (MVE) 指令集加速的图像均值融合 (HDR 简化版)。
 * 处理 8-bit 灰度图像。
 */
void helium_image_fusion(const uint8_t* img1, const uint8_t* img2, uint8_t* out, uint32_t num_pixels) {
    uint32_t blkCnt = num_pixels >> 4; // 每次处理 16 个像素 (128-bit 向量)
    
    const uint8_t* pIn1 = img1;
    const uint8_t* pIn2 = img2;
    uint8_t* pOut = out;

    while (blkCnt > 0U) {
        // 加载 16 个像素到向量寄存器
        uint8x16_t vecIn1 = vld1q_u8(pIn1);
        uint8x16_t vecIn2 = vld1q_u8(pIn2);
        
        // 向量加法并右移 1 位 (取平均)
        // vhaddq_u8 是 Helium 指令，执行 (a + b) >> 1
        uint8x16_t vecRes = vhaddq_u8(vecIn1, vecIn2);
        
        // 存储结果
        vst1q_u8(pOut, vecRes);
        
        pIn1 += 16;
        pIn2 += 16;
        pOut += 16;
        blkCnt--;
    }
    
    // 处理剩余像素
    uint32_t remainder = num_pixels & 0xF;
    for (uint32_t i = 0; i < remainder; i++) {
        pOut[i] = (uint8_t)(((uint16_t)pIn1[i] + (uint16_t)pIn2[i]) >> 1);
    }
}

/**
 * 简单的阈值处理，利用 Helium 加速。
 */
void helium_threshold(const uint8_t* src, uint8_t* dst, uint8_t threshold, uint32_t num_pixels) {
    uint32_t blkCnt = num_pixels >> 4;
    uint8x16_t vecThresh = vdupq_n_u8(threshold);
    uint8x16_t vecZero = vdupq_n_u8(0);
    uint8x16_t vecMax = vdupq_n_u8(255);

    while (blkCnt > 0U) {
        uint8x16_t vecIn = vld1q_u8(src);
        
        // 比较并创建掩码
        mve_pred16_t mask = vcmpgeq_u8(vecIn, vecThresh);
        
        // 根据掩码选择 0 或 255
        uint8x16_t vecRes = vpselq_u8(vecMax, vecZero, mask);
        
        vst1q_u8(dst, vecRes);
        
        src += 16;
        dst += 16;
        blkCnt--;
    }
}

// 注意：实际开发中需要包含 ARM 的 CMSIS-Core 和 CMSIS-DSP 库
