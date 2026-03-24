/**
 * @file operator_interface.h
 * @brief Unified operator execution interface for the EdgeVision-C AI vision architecture.
 * 
 * This file defines the function signatures for all vision operators,
 * supporting both reference and hardware-optimized implementations.
 * 
 * @copyright Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
 * @patent Protected by pending patent applications. See PATENTS for details.
 */

#ifndef EDGEVISION_C_OPERATOR_INTERFACE_H
#define EDGEVISION_C_OPERATOR_INTERFACE_H

#include "vision_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convolutional operator parameters.
 */
typedef struct {
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t padding_h;
    uint32_t padding_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
    int32_t activation_min;
    int32_t activation_max;
} ev_conv_params_t;

/**
 * @brief INT8 Convolution operator (Reference C Implementation).
 * 
 * @param input Input tensor (INT8)
 * @param filter Filter tensor (INT8)
 * @param bias Bias tensor (INT32)
 * @param output Output tensor (INT8)
 * @param params Convolution parameters
 * @return ev_status_t Status code
 */
ev_status_t ev_conv2d_int8_ref(
    const ev_tensor_t* input,
    const ev_tensor_t* filter,
    const ev_tensor_t* bias,
    ev_tensor_t* output,
    const ev_conv_params_t* params
);

/**
 * @brief INT8 Convolution operator (Arm Helium MVE Optimized).
 * 
 * @param input Input tensor (INT8)
 * @param filter Filter tensor (INT8)
 * @param bias Bias tensor (INT32)
 * @param output Output tensor (INT8)
 * @param params Convolution parameters
 * @return ev_status_t Status code
 */
ev_status_t ev_conv2d_int8_helium(
    const ev_tensor_t* input,
    const ev_tensor_t* filter,
    const ev_tensor_t* bias,
    ev_tensor_t* output,
    const ev_conv_params_t* params
);

#ifdef __cplusplus
}
#endif

#endif // EDGEVISION_C_OPERATOR_INTERFACE_H
