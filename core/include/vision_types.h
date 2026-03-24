/**
 * @file vision_types.h
 * @brief Core data structures for the EdgeVision-C AI vision architecture.
 * 
 * This file defines the fundamental types for tensors, quantization parameters,
 * and status codes used throughout the inference engine.
 * 
 * @copyright Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
 * @patent Protected by pending patent applications. See PATENTS for details.
 */

#ifndef EDGEVISION_C_VISION_TYPES_H
#define EDGEVISION_C_VISION_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes for the inference engine.
 */
typedef enum {
    EV_SUCCESS = 0,
    EV_ERROR_INVALID_ARGUMENT = -1,
    EV_ERROR_OUT_OF_MEMORY = -2,
    EV_ERROR_UNSUPPORTED_OPERATOR = -3,
    EV_ERROR_QUANTIZATION_MISMATCH = -4,
    EV_ERROR_DIMENSION_MISMATCH = -5,
    EV_ERROR_HARDWARE_FAILURE = -6
} ev_status_t;

/**
 * @brief Quantization parameters for INT8 tensors.
 * 
 * Based on the formula: RealValue = (QuantizedValue - ZeroPoint) * Scale
 */
typedef struct {
    float scale;           /**< Scaling factor for the tensor */
    int32_t zero_point;    /**< Zero point offset for the tensor */
} ev_quant_params_t;

/**
 * @brief Tensor data types.
 */
typedef enum {
    EV_TYPE_INT8 = 0,
    EV_TYPE_UINT8,
    EV_TYPE_INT16,
    EV_TYPE_INT32,
    EV_TYPE_FLOAT32
} ev_data_type_t;

/**
 * @brief Tensor structure for the EdgeVision-C engine.
 */
typedef struct {
    void* data;                    /**< Pointer to the raw tensor data */
    ev_data_type_t type;           /**< Data type of the tensor elements */
    uint32_t dims[4];              /**< Dimensions (N, H, W, C) */
    uint32_t num_dims;             /**< Number of dimensions */
    ev_quant_params_t quant;       /**< Quantization parameters */
    size_t data_size;              /**< Total size of the data in bytes */
    uint32_t memory_offset;        /**< Offset in the static memory pool */
} ev_tensor_t;

#ifdef __cplusplus
}
#endif

#endif // EDGEVISION_C_VISION_TYPES_H
