/**
 * @file operator_interface.h
 * @brief EdgeVision-C 统一算子接口
 *
 * 定义所有视觉算子的函数签名，支持参考实现和 Helium MVE 硬件加速实现。
 *
 * @copyright Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
 * @author  RussellCooper
 * @patent Protected by pending patent applications. See PATENTS for details.
 */

#ifndef EDGEVISION_C_OPERATOR_INTERFACE_H
#define EDGEVISION_C_OPERATOR_INTERFACE_H

#include <stdint.h>
#include <stddef.h>
#include "vision_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * 激活函数类型
 * ============================================================ */
typedef enum {
    EV_ACTIVATION_NONE  = 0,
    EV_ACTIVATION_RELU  = 1,
    EV_ACTIVATION_RELU6 = 2,
    EV_ACTIVATION_TANH  = 3,
} ev_activation_t;

/* ============================================================
 * 卷积算子参数
 * ============================================================ */
typedef struct {
    int32_t  stride_h;               /**< 高度方向步长 */
    int32_t  stride_w;               /**< 宽度方向步长 */
    int32_t  pad_top;                /**< 上侧 padding */
    int32_t  pad_bottom;             /**< 下侧 padding */
    int32_t  pad_left;               /**< 左侧 padding */
    int32_t  pad_right;              /**< 右侧 padding */
    int32_t  dilation_h;             /**< 高度方向空洞率（1=标准卷积）*/
    int32_t  dilation_w;             /**< 宽度方向空洞率 */
    int32_t  groups;                 /**< 分组数（1=标准，C_in=深度可分离）*/
    ev_activation_t activation;      /**< 激活函数类型 */
    /* 逐通道量化参数（由 QuantizeMultiplier 预计算，长度 = C_out）*/
    const int32_t* per_channel_multiplier;
    const int*     per_channel_shift;
} ev_conv_params_t;

/* ============================================================
 * INT8 标准卷积（参考实现，无 SIMD）
 * ============================================================ */

/**
 * @brief INT8 量化 2D 卷积参考实现（NHWC 格式）
 *
 * 支持任意 stride、padding、dilation、分组卷积、逐通道量化。
 *
 * @param input   输入张量 [N, H_in, W_in, C_in] INT8
 * @param filter  权重张量 [C_out, KH, KW, C_in/groups] INT8
 * @param bias    偏置张量 [C_out] INT32（可为 NULL）
 * @param output  输出张量 [N, H_out, W_out, C_out] INT8
 * @param params  卷积参数
 * @return ev_status_t 状态码
 */
ev_status_t ev_conv2d_int8_ref(
    const ev_tensor_t*      input,
    const ev_tensor_t*      filter,
    const ev_tensor_t*      bias,
    ev_tensor_t*            output,
    const ev_conv_params_t* params
);

/* ============================================================
 * INT8 卷积（Arm Helium MVE 加速版，RA8P1 专用）
 * ============================================================ */

/**
 * @brief INT8 量化 2D 卷积 Helium MVE 加速版
 *
 * 内层通道维度使用 vmladavaq 向量化，C_in 需为 16 的倍数，
 * 否则自动退化到参考实现。
 *
 * @param input   输入张量 [N, H_in, W_in, C_in] INT8
 * @param filter  权重张量 [C_out, KH, KW, C_in] INT8
 * @param bias    偏置张量 [C_out] INT32（可为 NULL）
 * @param output  输出张量 [N, H_out, W_out, C_out] INT8
 * @param params  卷积参数
 * @return ev_status_t 状态码
 */
ev_status_t ev_conv2d_int8_helium(
    const ev_tensor_t*      input,
    const ev_tensor_t*      filter,
    const ev_tensor_t*      bias,
    ev_tensor_t*            output,
    const ev_conv_params_t* params
);

/* ============================================================
 * INT8 深度可分离卷积（参考实现）
 * ============================================================ */

/**
 * @brief INT8 深度可分离卷积（Depthwise）参考实现
 *
 * filter 布局：[1, KH, KW, C_in * depth_multiplier]
 *
 * @param input   输入张量 [N, H_in, W_in, C_in] INT8
 * @param filter  深度卷积权重 [1, KH, KW, C_out] INT8
 * @param bias    偏置张量 [C_out] INT32（可为 NULL）
 * @param output  输出张量 [N, H_out, W_out, C_out] INT8
 * @param params  卷积参数（groups 字段忽略）
 * @return ev_status_t 状态码
 */
ev_status_t ev_depthwise_conv2d_int8_ref(
    const ev_tensor_t*      input,
    const ev_tensor_t*      filter,
    const ev_tensor_t*      bias,
    ev_tensor_t*            output,
    const ev_conv_params_t* params
);

/* ============================================================
 * INT8 全连接层（参考实现）
 * ============================================================ */

/**
 * @brief INT8 全连接层参考实现
 *
 * @param input    输入张量 [batch, input_size] INT8
 * @param weights  权重张量 [output_size, input_size] INT8
 * @param bias     偏置张量 [output_size] INT32（可为 NULL）
 * @param output   输出张量 [batch, output_size] INT8
 * @param params   参数（activation 字段有效）
 * @return ev_status_t 状态码
 */
ev_status_t ev_fully_connected_int8_ref(
    const ev_tensor_t*      input,
    const ev_tensor_t*      weights,
    const ev_tensor_t*      bias,
    ev_tensor_t*            output,
    const ev_conv_params_t* params
);

/* ============================================================
 * INT8 平均池化（参考实现）
 * ============================================================ */

/**
 * @brief INT8 平均池化参考实现
 *
 * 注意：复用 ev_conv_params_t 字段：
 *   stride_h/stride_w = 池化核大小
 *   pad_top/pad_left  = 池化步长
 *
 * @param input   输入张量 [N, H_in, W_in, C] INT8
 * @param output  输出张量 [N, H_out, W_out, C] INT8
 * @param params  池化参数
 * @return ev_status_t 状态码
 */
ev_status_t ev_avg_pool2d_int8_ref(
    const ev_tensor_t*      input,
    ev_tensor_t*            output,
    const ev_conv_params_t* params
);

/* ============================================================
 * 工业级视觉算子 (针对高反光工件优化)
 * ============================================================ */

/**
 * @brief HDR 曝光融合 (Helium 向量化实现)
 * 融合欠曝、正常、过曝三帧图像以消除反光。
 */
ev_status_t ev_hdr_fusion_helium(
    const uint8_t* under_exp,
    const uint8_t* normal_exp,
    const uint8_t* over_exp,
    uint8_t*       output,
    uint32_t       width,
    uint32_t       height
);

/**
 * @brief 高光抑制滤波器
 * 基于自适应阈值和 Helium 加速的中值/均值滤波。
 */
ev_status_t ev_glare_suppression_helium(
    const uint8_t* input,
    uint8_t*       output,
    uint32_t       width,
    uint32_t       height,
    uint8_t        threshold
);

/* ============================================================
 * 静态内存池管理
 * ============================================================ */

/**
 * @brief 初始化静态内存池
 *
 * 必须在调用任何 ev_memory_pool_alloc 之前调用。
 *
 * @param buffer  外部提供的静态缓冲区（全局数组）
 * @param size    缓冲区大小（字节）
 */
void ev_memory_pool_init(uint8_t* buffer, size_t size);

/**
 * @brief 从内存池分配对齐内存
 *
 * @param size       请求字节数
 * @param alignment  对齐字节数（必须是 2 的幂，推荐 16）
 * @return 分配的内存指针，内存不足时返回 NULL
 */
void* ev_memory_pool_alloc(size_t size, size_t alignment);

/**
 * @brief 重置内存池（释放所有分配，O(1) 操作）
 */
void ev_memory_pool_reset(void);

/**
 * @brief 查询内存池剩余可用空间（字节）
 */
size_t ev_memory_pool_available(void);

#ifdef __cplusplus
}
#endif

#endif /* EDGEVISION_C_OPERATOR_INTERFACE_H */
