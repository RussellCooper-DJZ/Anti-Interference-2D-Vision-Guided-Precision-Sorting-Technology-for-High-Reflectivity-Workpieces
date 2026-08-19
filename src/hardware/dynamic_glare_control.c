/**
 * @file dynamic_glare_control.c
 * @brief 针对高反光工件的动态曝光与增益控制策略
 * 
 * 该模块根据图像的实时直方图反馈，动态调整 IMX 传感器的寄存器，
 * 以实现对不同材质（不锈钢、铝合金、电镀件）的自适应识别。
 */

#include "imx_reg_helper.c"
#include <math.h>

typedef enum {
    MATERIAL_STAINLESS_STEEL,
    MATERIAL_ALUMINUM,
    MATERIAL_ELECTROPLATED,
    MATERIAL_UNKNOWN
} workpiece_material_t;

/**
 * @brief 自动调整传感器参数以应对特定材质的高反光
 */
void adjust_sensor_for_material(workpiece_material_t material) {
    switch (material) {
        case MATERIAL_STAINLESS_STEEL:
            /* 不锈钢：中等反射，高对比度 */
            imx_optimize_gain_for_glare(0);    // 0dB 模拟增益
            imx_enable_dol_hdr(true);         // 开启 HDR 捕获边缘
            break;
            
        case MATERIAL_ALUMINUM:
            /* 铝合金：漫反射较多，高光区域大 */
            imx_optimize_gain_for_glare(3);    // 适当提升增益以看清暗部
            imx_set_black_level(15);          // 提高黑电平压制背景噪声
            break;
            
        case MATERIAL_ELECTROPLATED:
            /* 电镀件：镜面反射极强，极易过曝 */
            imx_optimize_gain_for_glare(0);    // 强制 0dB
            imx_enable_dol_hdr(true);
            // 触发超短曝光序列 (e.g. 100us)
            imx_trigger_exposure_sequence(100, 5000); 
            break;
            
        default:
            imx_enable_dol_hdr(false);
            break;
    }
}

/**
 * @brief 基于直方图的实时高光压制闭环控制
 * 
 * 由 RA8P1 定时器或每帧处理完成后调用。
 */
void glare_control_feedback_loop(uint32_t overexposed_pixel_count) {
    static uint32_t current_exposure = 5000; // us
    
    // 如果过曝像素占比超过 5%，则降低曝光时间
    if (overexposed_pixel_count > (IMAGE_WIDTH * IMAGE_HEIGHT * 0.05)) {
        current_exposure = (uint32_t)(current_exposure * 0.8);
        if (current_exposure < 50) current_exposure = 50;
        
        // 更新 IMX 寄存器
        // imx_write_reg16(IMX335_REG_SHUTTER, current_exposure);
    }
}
