#include "hal_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h" // 包含转换后的模型数组

// 图像缓冲区定义 (根据实际分辨率调整)
#define IMG_WIDTH  320
#define IMG_HEIGHT 240
uint8_t g_img_buffer_under[IMG_WIDTH * IMG_HEIGHT];
uint8_t g_img_buffer_over[IMG_WIDTH * IMG_HEIGHT];
uint8_t g_img_buffer_hdr[IMG_WIDTH * IMG_HEIGHT];
uint8_t g_mask_buffer[IMG_WIDTH * IMG_HEIGHT];

// TFLM 相关定义
namespace {
    const int kTensorArenaSize = 128 * 1024; // 128KB 内存池
    uint8_t tensor_arena[kTensorArenaSize];
}

void hal_entry(void) {
    /* 初始化 FSP 驱动 (CSI, I2C, UART 等) */
    R_CSI_Open(&g_csi0_ctrl, &g_csi0_cfg);
    R_SCI_UART_Open(&g_uart0_ctrl, &g_uart0_cfg);
    
    /* 初始化 TFLM 解释器 */
    static tflite::MicroInterpreter interpreter(
        tflite::GetModel(g_model_data), resolver, tensor_arena, kTensorArenaSize);
    interpreter.AllocateTensors();
    
    while (1) {
        /* 1. 采集多重曝光图像 */
        // 伪代码：控制传感器曝光并捕获图像到 g_img_buffer_under 和 g_img_buffer_over
        capture_multi_exposure(g_img_buffer_under, g_img_buffer_over);
        
        /* 2. HDR 融合 (Helium 加速) */
        helium_image_fusion(g_img_buffer_under, g_img_buffer_over, g_img_buffer_hdr, IMG_WIDTH * IMG_HEIGHT);
        
        /* 3. 深度学习推理 */
        // 将 g_img_buffer_hdr 数据拷贝到模型输入 tensor
        int8_t* input = interpreter.input(0)->data.int8;
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
            input[i] = (int8_t)((int16_t)g_img_buffer_hdr[i] - 128); // 简单的归一化到 int8
        }
        
        interpreter.Invoke();
        
        // 获取输出掩膜
        int8_t* output = interpreter.output(0)->data.int8;
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
            g_mask_buffer[i] = (output[i] > 0) ? 255 : 0;
        }
        
        /* 4. 亚像素定位与坐标转换 */
        float x, y, theta;
        if (calculate_localization(g_mask_buffer, &x, &y, &theta)) {
            /* 5. 发送结果到机器人 */
            char msg[64];
            sprintf(msg, "X:%.2f, Y:%.2f, T:%.2f\r\n", x, y, theta);
            R_SCI_UART_Write(&g_uart0_ctrl, (uint8_t*)msg, strlen(msg));
        }
        
        R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MILLISECONDS);
    }
}
