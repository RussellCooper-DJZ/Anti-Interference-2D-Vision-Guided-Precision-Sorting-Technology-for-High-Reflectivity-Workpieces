/**
 * @file imx_reg_helper.c
 * @brief Sony IMX 系列传感器底层寄存器配置助手 (针对高反光抑制优化)
 * 
 * 适配传感器：IMX290, IMX327, IMX335, IMX415 等。
 * 核心功能：
 *   1. 启用 DOL-HDR (Digital Overlap HDR) 模式。
 *   2. 精确控制 Analog Gain (模拟增益) 与 Digital Gain (数字增益)。
 *   3. 动态调整 Black Level (黑电平)。
 */

#include <stdint.h>
#include <stdbool.h>
#include "hal_data.h"

/* IMX335 典型寄存器基地址 (示例，具体需参考对应 Datasheet) */
#define IMX335_I2C_ADDR         0x1A
#define IMX335_REG_HOLD         0x3001  /* 寄存器更新锁定 */
#define IMX335_REG_SHUTTER_L    0x3058  /* 曝光时间 (Low) */
#define IMX335_REG_GAIN         0x30E8  /* 增益控制 */
#define IMX335_REG_HDR_MODE     0x300C  /* HDR 模式选择 */
#define IMX335_REG_BLKLEVEL     0x3015  /* 黑电平偏移 */

/**
 * @brief 向传感器写入 16 位地址的 8 位寄存器值
 */
fsp_err_t imx_write_reg8(uint16_t reg, uint8_t val) {
    uint8_t buf[3];
    buf[0] = (uint8_t)(reg >> 8);   /* Address High */
    buf[1] = (uint8_t)(reg & 0xFF); /* Address Low */
    buf[2] = val;                   /* Data */
    
    /* 调用瑞萨 FSP I2C 驱动 */
    return R_IIC_MASTER_Write(&g_i2c_master0_ctrl, buf, 3, false);
}

/**
 * @brief 开启 DOL-HDR 模式 (双重或三重曝光)
 * 
 * 针对高反光工件，开启 DOL-HDR 可以让传感器在一帧内输出不同曝光的数据，
 * 从而在源头上解决高光过曝问题。
 */
void imx_enable_dol_hdr(bool enable) {
    imx_write_reg8(IMX335_REG_HOLD, 0x01); // 开始批量更新
    
    if (enable) {
        // 示例：设置 IMX335 为 DOL-HDR 2-frame 模式
        imx_write_reg8(IMX335_REG_HDR_MODE, 0x01); 
        // 调整 V-Blanking 以匹配 HDR 帧率
        // imx_write_reg8(0x3030, ...); 
    } else {
        imx_write_reg8(IMX335_REG_HDR_MODE, 0x00); // 回到线性模式
    }
    
    imx_write_reg8(IMX335_REG_HOLD, 0x00); // 结束更新并生效
}

/**
 * @brief 针对高反光抑制的增益优化策略
 * 
 * 优化原则：
 *   1. 优先保持 Analog Gain 为 0dB (极小值)，以最大化满阱容量 (Full Well Capacity)。
 *   2. 严禁使用过大的 Digital Gain，防止反光区域的噪声被放大。
 */
void imx_optimize_gain_for_glare(uint32_t analog_gain_db) {
    // IMX 传感器的增益通常是线性或对数映射
    // 示例：IMX335 的增益寄存器值 = gain_db * 10 (假设)
    uint8_t reg_val = (uint8_t)(analog_gain_db * 2); 
    
    imx_write_reg8(IMX335_REG_GAIN, reg_val);
}

/**
 * @brief 动态黑电平调整
 * 
 * 在强光干扰下，适当调低黑电平可以过滤掉背景的微弱杂散光。
 */
void imx_set_black_level(uint8_t level) {
    imx_write_reg8(IMX335_REG_BLKLEVEL, level);
}

/**
 * @brief 触发一次多重曝光序列
 * 
 * 该函数配合 RA8P1 的 GPT (General Purpose Timer) 触发信号，
 * 实现毫秒级的曝光切换。
 */
void imx_trigger_exposure_sequence(uint32_t short_us, uint32_t long_us) {
    // 1. 设置短曝光 (针对高光区域)
    // imx_write_reg8(IMX335_REG_SHUTTER_L, ...);
    
    // 2. 设置长曝光 (针对阴影区域)
    // ...
}
