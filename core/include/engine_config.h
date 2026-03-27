/**
 * @file engine_config.h
 * @brief Static configuration for the EdgeVision-C AI vision architecture.
 * 
 * This file defines the static memory pool size and operator switches
 * for the inference engine.
 * 
 * @copyright Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
 * @author  RussellCooper
 * @patent Protected by pending patent applications. See PATENTS for details.
 */

#ifndef EDGEVISION_C_ENGINE_CONFIG_H
#define EDGEVISION_C_ENGINE_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Static memory pool size in bytes.
 * 
 * This pool is used for all tensor allocations during inference.
 * No dynamic memory allocation (malloc/free) is allowed.
 */
#define EV_STATIC_MEMORY_POOL_SIZE (1024 * 1024) // 1MB

/**
 * @brief Operator switches for code size optimization.
 */
#define EV_ENABLE_CONV2D_INT8 1
#define EV_ENABLE_RELU_INT8 1
#define EV_ENABLE_POOLING_INT8 1
#define EV_ENABLE_SOFTMAX_INT8 1

/**
 * @brief Hardware acceleration switches.
 */
#define EV_USE_ARM_HELIUM_MVE 1
#define EV_USE_ARM_DSP_INTRINSICS 0

#ifdef __cplusplus
}
#endif

#endif // EDGEVISION_C_ENGINE_CONFIG_H
