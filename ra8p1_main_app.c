#include "hal_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h" // 包含转换后的模型数组
// Includes the converted model array
// Enthält das konvertierte Modellarray

// 图像缓冲区定义 (根据实际分辨率调整)
// Image buffer definitions (adjust according to actual resolution)
// Bildpufferdefinitionen (entsprechend der tatsächlichen Auflösung anpassen)
#define IMG_WIDTH  320
#define IMG_HEIGHT 240
uint8_t g_img_buffer_under[IMG_WIDTH * IMG_HEIGHT];
uint8_t g_img_buffer_over[IMG_WIDTH * IMG_HEIGHT];
uint8_t g_img_buffer_hdr[IMG_WIDTH * IMG_HEIGHT];
uint8_t g_mask_buffer[IMG_WIDTH * IMG_HEIGHT];

// TFLM 相关定义
// TFLM related definitions
// TFLM-bezogene Definitionen
namespace {
    const int kTensorArenaSize = 128 * 1024; // 128KB 内存池
    // 128KB memory pool
    // 128KB Speicherpool
    uint8_t tensor_arena[kTensorArenaSize];
}

void hal_entry(void) {
    /* 初始化 FSP 驱动 (CSI, I2C, UART 等) */
    /* Initialize FSP drivers (CSI, I2C, UART, etc.) */
    /* FSP-Treiber initialisieren (CSI, I2C, UART usw.) */
    R_CSI_Open(&g_csi0_ctrl, &g_csi0_cfg);
    R_SCI_UART_Open(&g_uart0_ctrl, &g_uart0_cfg);
    
    /* 初始化 TFLM 解释器 */
    /* Initialize TFLM interpreter */
    /* TFLM-Interpreter initialisieren */
    static tflite::MicroInterpreter interpreter(
        tflite::GetModel(g_model_data), resolver, tensor_arena, kTensorArenaSize);
    interpreter.AllocateTensors();
    
    while (1) {
        /* 1. 采集多重曝光图像 */
        /* 1. Acquire multi-exposure images */
        /* 1. Mehrfachbelichtungsbilder erfassen */
        // 伪代码：控制传感器曝光并捕获图像到 g_img_buffer_under 和 g_img_buffer_over
        // Pseudocode: Control sensor exposure and capture images to g_img_buffer_under and g_img_buffer_over
        // Pseudocode: Sensorbelichtung steuern und Bilder in g_img_buffer_under und g_img_buffer_over erfassen
        capture_multi_exposure(g_img_buffer_under, g_img_buffer_over);
        
        /* 2. HDR 融合 (Helium 加速) */
        /* 2. HDR Fusion (Helium accelerated) */
        /* 2. HDR-Fusion (Helium-beschleunigt) */
        helium_image_fusion(g_img_buffer_under, g_img_buffer_over, g_img_buffer_hdr, IMG_WIDTH * IMG_HEIGHT);
        
        /* 3. 深度学习推理 */
        /* 3. Deep Learning Inference */
        /* 3. Deep-Learning-Inferenz */
        // 将 g_img_buffer_hdr 数据拷贝到模型输入 tensor
        // Copy g_img_buffer_hdr data to model input tensor
        // g_img_buffer_hdr-Daten in den Modelleingabetensor kopieren
        int8_t* input = interpreter.input(0)->data.int8;
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
            input[i] = (int8_t)((int16_t)g_img_buffer_hdr[i] - 128); // 简单的归一化到 int8
            // Simple normalization to int8
            // Einfache Normalisierung auf int8
        }
        
        interpreter.Invoke();
        
        // 获取输出掩膜
        // Get output mask
        // Ausgabemaske erhalten
        int8_t* output = interpreter.output(0)->data.int8;
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
            g_mask_buffer[i] = (output[i] > 0) ? 255 : 0;
        }
        
        /* 4. 亚像素定位与坐标转换 */
        /* 4. Sub-pixel localization and coordinate transformation */
        /* 4. Subpixel-Lokalisierung und Koordinatentransformation */
        float x, y, theta;
        if (calculate_localization(g_mask_buffer, &x, &y, &theta)) {
            /* 5. 发送结果到机器人 */
            /* 5. Send results to robot */
            /* 5. Ergebnisse an Roboter senden */
            char msg[64];
            sprintf(msg, "X:%.2f, Y:%.2f, T:%.2f\r\n", x, y, theta);
            R_SCI_UART_Write(&g_uart0_ctrl, (uint8_t*)msg, strlen(msg));
        }
        
        R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MILLISECONDS);
    }
}
