/**
 * @file fsp_mock.c
 * @brief 瑞萨 FSP 驱动模拟桩，用于在 PC 上验证业务逻辑。
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

typedef int fsp_err_t;
#define FSP_SUCCESS 0

void R_IOPORT_Open(void* c, void* cfg) { printf("[Mock] IOPORT Opened\n"); }
void R_CSI_Open(void* c, void* cfg) { printf("[Mock] CSI-2 Opened\n"); }
void R_SCI_UART_Open(void* c, void* cfg) { printf("[Mock] UART Opened\n"); }
void R_SCI_UART_Write(void* c, uint8_t* data, size_t len) {
    printf("[Mock] UART Send: %.*s", (int)len, data);
}

void capture_multi_exposure(uint8_t* u, uint8_t* n, uint8_t* o) {
    memset(u, 50, 320*240);
    memset(n, 128, 320*240);
    memset(o, 200, 320*240);
    printf("[Mock] Multi-exposure captured\n");
}
