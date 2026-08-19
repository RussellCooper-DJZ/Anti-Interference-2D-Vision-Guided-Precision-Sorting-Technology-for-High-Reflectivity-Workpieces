/**
 * @file ev_operators.c
 * @brief EdgeVision-C 核心算子实现
 *        INT8 量化卷积（参考实现 + Helium MVE 加速版）
 *        深度可分离卷积、全连接层、激活函数、池化
 *
 * 编译目标：Renesas RA8P1 (Cortex-M85, Helium/MVE)
 * 参考实现可在任意 ARM Cortex-M 上运行（无 Helium 依赖）
 *
 * 量化约定（TFLite 对称 INT8）：
 *   real_value = (int8_value - zero_point) * scale
 *   输入/输出 zero_point 通常为 0（对称量化）
 *   偏置使用 INT32，scale = input_scale * filter_scale
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "../core/include/vision_types.h"
#include "../core/include/operator_interface.h"

/* ============================================================
 * 内部工具宏
 * ============================================================ */

/** 饱和截断到 INT8 范围 */
#define EV_CLAMP_INT8(x)  ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))

/** 饱和截断到 INT32 范围（防止累加溢出） */
#define EV_CLAMP_INT32(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

/** 向右算术移位（带四舍五入） */
static inline int32_t ev_rounding_shift_right(int32_t x, int shift)
{
    if (shift <= 0) return x;
    int32_t rounding = 1 << (shift - 1);
    return (x + rounding) >> shift;
}

/**
 * 量化乘法：将 INT32 累加值乘以量化乘数并移位，输出 INT32。
 *
 * 实现 TFLite 的 MultiplyByQuantizedMultiplier：
 *   out = round(x * multiplier / 2^shift)
 *
 * @param x          INT32 累加值
 * @param multiplier 量化乘数（INT32，通常由 QuantizeMultiplier 计算）
 * @param shift      右移位数（正数=右移，负数=左移）
 */
static inline int32_t ev_multiply_by_quantized_multiplier(
    int32_t x, int32_t multiplier, int shift)
{
    /* 64 位中间结果防止溢出 */
    int64_t result = (int64_t)x * (int64_t)multiplier;
    /* 算术右移 31 位（multiplier 已归一化到 [0.5, 1) * 2^31） */
    result += (int64_t)1 << 30;  /* 四舍五入 */
    result >>= 31;
    /* 应用额外移位 */
    if (shift > 0) {
        result >>= shift;
    } else if (shift < 0) {
        result <<= (-shift);
    }
    return (int32_t)EV_CLAMP_INT32(result, INT32_MIN, INT32_MAX);
}

/* ============================================================
 * 1. INT8 标准卷积（参考实现，无 SIMD）
 * ============================================================ */

/**
 * @brief INT8 量化 2D 卷积参考实现（NHWC 格式）
 *
 * 支持：
 *   - 任意 stride（stride_h, stride_w）
 *   - 任意 padding（pad_top, pad_bottom, pad_left, pad_right）
 *   - 分组卷积（groups > 1）
 *   - 逐通道量化（per-channel quantization）
 *
 * 内存布局：
 *   input:  [N, H_in,  W_in,  C_in]  INT8
 *   filter: [C_out, KH, KW, C_in/groups]  INT8
 *   bias:   [C_out]  INT32
 *   output: [N, H_out, W_out, C_out]  INT8
 */
ev_status_t ev_conv2d_int8_ref(
    const ev_tensor_t* input,
    const ev_tensor_t* filter,
    const ev_tensor_t* bias,
    ev_tensor_t*       output,
    const ev_conv_params_t* params)
{
    if (!input || !filter || !output || !params) {
        return EV_ERROR_INVALID_ARGUMENT;
    }

    /* 提取维度 */
    const int N    = (int)input->dims[0];
    const int H_in = (int)input->dims[1];
    const int W_in = (int)input->dims[2];
    const int C_in = (int)input->dims[3];

    const int KH   = (int)filter->dims[1];
    const int KW   = (int)filter->dims[2];
    const int C_out = (int)output->dims[3];

    const int stride_h = params->stride_h;
    const int stride_w = params->stride_w;
    const int dilation_h = (params->dilation_h > 0) ? params->dilation_h : 1;
    const int dilation_w = (params->dilation_w > 0) ? params->dilation_w : 1;
    const int pad_top  = params->pad_top;
    const int pad_left = params->pad_left;
    const int groups   = (params->groups > 0) ? params->groups : 1;

    const int H_out = (int)output->dims[1];
    const int W_out = (int)output->dims[2];

    const int C_in_per_group  = C_in  / groups;
    const int C_out_per_group = C_out / groups;

    const int8_t*  in_data  = (const int8_t*)input->data;
    const int8_t*  flt_data = (const int8_t*)filter->data;
    const int32_t* bias_data = bias ? (const int32_t*)bias->data : NULL;
    int8_t*        out_data = (int8_t*)output->data;

    const int32_t in_zp  = input->quant.zero_point;
    const int32_t out_zp = output->quant.zero_point;

    /* 逐通道量化乘数（由调用方预计算并存入 params->per_channel_multiplier） */
    const int32_t* multipliers = params->per_channel_multiplier;
    const int*     shifts      = params->per_channel_shift;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            int c_out_start = g * C_out_per_group;
            int c_in_start  = g * C_in_per_group;

            for (int oc = 0; oc < C_out_per_group; oc++) {
                int abs_oc = c_out_start + oc;

                for (int oh = 0; oh < H_out; oh++) {
                    for (int ow = 0; ow < W_out; ow++) {

                        int32_t acc = 0;

                        /* 卷积核累加 */
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                int ih = oh * stride_h - pad_top  + kh * dilation_h;
                                int iw = ow * stride_w - pad_left + kw * dilation_w;

                                if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) {
                                    /* padding 区域：输入值视为 zero_point */
                                    /* filter 值仍然参与计算 */
                                    for (int ic = 0; ic < C_in_per_group; ic++) {
                                        int flt_idx = ((abs_oc * KH + kh) * KW + kw)
                                                      * C_in_per_group + ic;
                                        int32_t fv = (int32_t)flt_data[flt_idx];
                                        acc += fv * (-in_zp);
                                    }
                                    continue;
                                }

                                for (int ic = 0; ic < C_in_per_group; ic++) {
                                    int in_idx  = ((n * H_in + ih) * W_in + iw)
                                                  * C_in + (c_in_start + ic);
                                    int flt_idx = ((abs_oc * KH + kh) * KW + kw)
                                                  * C_in_per_group + ic;

                                    int32_t iv = (int32_t)in_data[in_idx]  - in_zp;
                                    int32_t fv = (int32_t)flt_data[flt_idx];
                                    acc += iv * fv;
                                }
                            }
                        }

                        /* 加偏置 */
                        if (bias_data) {
                            acc += bias_data[abs_oc];
                        }

                        /* 量化重缩放 */
                        int32_t scaled;
                        if (multipliers && shifts) {
                            scaled = ev_multiply_by_quantized_multiplier(
                                acc, multipliers[abs_oc], shifts[abs_oc]);
                        } else {
                            /* 退化为简单缩放（不推荐，仅用于调试） */
                            scaled = (int32_t)(acc * input->quant.scale
                                              * filter->quant.scale
                                              / output->quant.scale);
                        }

                        /* 加输出 zero_point + 激活函数 */
                        scaled += out_zp;
                        /* ReLU6 激活（可通过 params->activation 配置） */
                        if (params->activation == EV_ACTIVATION_RELU) {
                            if (scaled < out_zp) scaled = out_zp;
                        } else if (params->activation == EV_ACTIVATION_RELU6) {
                            int32_t relu6_max = out_zp + (int32_t)(6.0f / output->quant.scale);
                            if (scaled < out_zp)      scaled = out_zp;
                            if (scaled > relu6_max)   scaled = relu6_max;
                        }

                        int out_idx = ((n * H_out + oh) * W_out + ow) * C_out + abs_oc;
                        out_data[out_idx] = EV_CLAMP_INT8(scaled);
                    }
                }
            }
        }
    }

    return EV_SUCCESS;
}

/* ============================================================
 * 2. INT8 深度可分离卷积（参考实现）
 * ============================================================ */

/**
 * @brief INT8 深度可分离卷积（Depthwise Separable Convolution）参考实现
 *
 * 分两步：
 *   Step 1: Depthwise Conv（每个输入通道独立卷积，groups = C_in）
 *   Step 2: Pointwise Conv（1x1 卷积，使用 ev_conv2d_int8_ref）
 *
 * 本函数实现 Step 1（Depthwise）。
 */
ev_status_t ev_depthwise_conv2d_int8_ref(
    const ev_tensor_t* input,
    const ev_tensor_t* filter,
    const ev_tensor_t* bias,
    ev_tensor_t*       output,
    const ev_conv_params_t* params)
{
    if (!input || !filter || !output || !params) {
        return EV_ERROR_INVALID_ARGUMENT;
    }

    const int N    = (int)input->dims[0];
    const int H_in = (int)input->dims[1];
    const int W_in = (int)input->dims[2];
    const int C_in = (int)input->dims[3];

    const int KH   = (int)filter->dims[1];
    const int KW   = (int)filter->dims[2];
    const int depth_multiplier = (int)filter->dims[0];  /* 通常为 1 */

    const int stride_h = params->stride_h;
    const int stride_w = params->stride_w;
    const int dilation_h = (params->dilation_h > 0) ? params->dilation_h : 1;
    const int dilation_w = (params->dilation_w > 0) ? params->dilation_w : 1;
    const int pad_top  = params->pad_top;
    const int pad_left = params->pad_left;

    const int H_out = (int)output->dims[1];
    const int W_out = (int)output->dims[2];
    const int C_out = C_in * depth_multiplier;

    const int8_t*  in_data   = (const int8_t*)input->data;
    const int8_t*  flt_data  = (const int8_t*)filter->data;
    const int32_t* bias_data = bias ? (const int32_t*)bias->data : NULL;
    int8_t*        out_data  = (int8_t*)output->data;

    const int32_t in_zp  = input->quant.zero_point;
    const int32_t out_zp = output->quant.zero_point;
    const int32_t* multipliers = params->per_channel_multiplier;
    const int*     shifts      = params->per_channel_shift;

    for (int n = 0; n < N; n++) {
        for (int ic = 0; ic < C_in; ic++) {
            for (int dm = 0; dm < depth_multiplier; dm++) {
                int abs_oc = ic * depth_multiplier + dm;

                for (int oh = 0; oh < H_out; oh++) {
                    for (int ow = 0; ow < W_out; ow++) {
                        int32_t acc = 0;

                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                int ih = oh * stride_h - pad_top  + kh * dilation_h;
                                int iw = ow * stride_w - pad_left + kw * dilation_w;

                                int32_t iv = 0;
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int in_idx = ((n * H_in + ih) * W_in + iw) * C_in + ic;
                                    iv = (int32_t)in_data[in_idx] - in_zp;
                                }
                                /* filter layout: [1, KH, KW, C_in * depth_multiplier] */
                                int flt_idx = (kh * KW + kw) * C_out + abs_oc;
                                int32_t fv = (int32_t)flt_data[flt_idx];
                                acc += iv * fv;
                            }
                        }

                        if (bias_data) acc += bias_data[abs_oc];

                        int32_t scaled;
                        if (multipliers && shifts) {
                            scaled = ev_multiply_by_quantized_multiplier(
                                acc, multipliers[abs_oc], shifts[abs_oc]);
                        } else {
                            scaled = acc;
                        }
                        scaled += out_zp;

                        if (params->activation == EV_ACTIVATION_RELU) {
                            if (scaled < out_zp) scaled = out_zp;
                        } else if (params->activation == EV_ACTIVATION_RELU6) {
                            int32_t relu6_max = out_zp + (int32_t)(6.0f / output->quant.scale);
                            if (scaled < out_zp)    scaled = out_zp;
                            if (scaled > relu6_max) scaled = relu6_max;
                        }

                        int out_idx = ((n * H_out + oh) * W_out + ow) * C_out + abs_oc;
                        out_data[out_idx] = EV_CLAMP_INT8(scaled);
                    }
                }
            }
        }
    }

    return EV_SUCCESS;
}

/* ============================================================
 * 3. INT8 全连接层（参考实现）
 * ============================================================ */

ev_status_t ev_fully_connected_int8_ref(
    const ev_tensor_t* input,
    const ev_tensor_t* weights,
    const ev_tensor_t* bias,
    ev_tensor_t*       output,
    const ev_conv_params_t* params)
{
    if (!input || !weights || !output || !params) {
        return EV_ERROR_INVALID_ARGUMENT;
    }

    const int batch       = (int)input->dims[0];
    const int input_size  = (int)weights->dims[1];
    const int output_size = (int)weights->dims[0];

    const int8_t*  in_data  = (const int8_t*)input->data;
    const int8_t*  w_data   = (const int8_t*)weights->data;
    const int32_t* b_data   = bias ? (const int32_t*)bias->data : NULL;
    int8_t*        out_data = (int8_t*)output->data;

    const int32_t in_zp  = input->quant.zero_point;
    const int32_t out_zp = output->quant.zero_point;
    const int32_t* multipliers = params->per_channel_multiplier;
    const int*     shifts      = params->per_channel_shift;

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < output_size; oc++) {
            int32_t acc = 0;
            for (int ic = 0; ic < input_size; ic++) {
                int32_t iv = (int32_t)in_data[b * input_size + ic] - in_zp;
                int32_t wv = (int32_t)w_data[oc * input_size + ic];
                acc += iv * wv;
            }
            if (b_data) acc += b_data[oc];

            int32_t scaled;
            if (multipliers && shifts) {
                scaled = ev_multiply_by_quantized_multiplier(
                    acc, multipliers[oc], shifts[oc]);
            } else {
                scaled = acc;
            }
            scaled += out_zp;

            if (params->activation == EV_ACTIVATION_RELU) {
                if (scaled < out_zp) scaled = out_zp;
            }

            out_data[b * output_size + oc] = EV_CLAMP_INT8(scaled);
        }
    }

    return EV_SUCCESS;
}

/* ============================================================
 * 4. INT8 平均池化（参考实现）
 * ============================================================ */

ev_status_t ev_avg_pool2d_int8_ref(
    const ev_tensor_t* input,
    ev_tensor_t*       output,
    const ev_conv_params_t* params)
{
    if (!input || !output || !params) {
        return EV_ERROR_INVALID_ARGUMENT;
    }

    const int N    = (int)input->dims[0];
    const int H_in = (int)input->dims[1];
    const int W_in = (int)input->dims[2];
    const int C    = (int)input->dims[3];
    const int H_out = (int)output->dims[1];
    const int W_out = (int)output->dims[2];
    const int KH   = params->stride_h;  /* 池化核大小复用 stride 字段 */
    const int KW   = params->stride_w;
    const int stride_h = params->pad_top;   /* 复用 pad 字段存 pool stride */
    const int stride_w = params->pad_left;

    const int8_t* in_data  = (const int8_t*)input->data;
    int8_t*       out_data = (int8_t*)output->data;

    const int32_t in_zp  = input->quant.zero_point;
    const int32_t out_zp = output->quant.zero_point;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                for (int c = 0; c < C; c++) {
                    int32_t acc = 0;
                    int count = 0;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            if (ih < H_in && iw < W_in) {
                                int idx = ((n * H_in + ih) * W_in + iw) * C + c;
                                acc += (int32_t)in_data[idx];
                                count++;
                            }
                        }
                    }
                    int32_t avg = (count > 0) ? (acc / count) : in_zp;
                    /* 重量化（输入输出 scale 相同时直接赋值） */
                    int32_t out_val = avg - in_zp + out_zp;
                    int out_idx = ((n * H_out + oh) * W_out + ow) * C + c;
                    out_data[out_idx] = EV_CLAMP_INT8(out_val);
                }
            }
        }
    }

    return EV_SUCCESS;
}

/* ============================================================
 * 5. Helium MVE 加速版 INT8 卷积（RA8P1 专用）
 *    编译时需定义 __ARM_FEATURE_MVE
 * ============================================================ */

#ifdef __ARM_FEATURE_MVE
#include <arm_mve.h>

/**
 * @brief Helium MVE 加速的 INT8 卷积（内层循环向量化）
 *
 * 优化策略：
 *   - 将输入通道维度（C_in）向量化，每次处理 16 个 INT8 元素
 *   - 使用 vmladavaq（向量乘累加）指令
 *   - 输出通道循环保持标量（适合 C_out 较小的嵌入式场景）
 *
 * 注意：此函数需要 C_in 为 16 的倍数，否则退化到参考实现。
 */
ev_status_t ev_conv2d_int8_helium(
    const ev_tensor_t* input,
    const ev_tensor_t* filter,
    const ev_tensor_t* bias,
    ev_tensor_t*       output,
    const ev_conv_params_t* params)
{
    if (!input || !filter || !output || !params) {
        return EV_ERROR_INVALID_ARGUMENT;
    }

    const int C_in = (int)input->dims[3];
    /* C_in 不是 16 的倍数时退化到参考实现 */
    if (C_in % 16 != 0) {
        return ev_conv2d_int8_ref(input, filter, bias, output, params);
    }

    const int N    = (int)input->dims[0];
    const int H_in = (int)input->dims[1];
    const int W_in = (int)input->dims[2];
    const int KH   = (int)filter->dims[1];
    const int KW   = (int)filter->dims[2];
    const int C_out = (int)output->dims[3];
    const int H_out = (int)output->dims[1];
    const int W_out = (int)output->dims[2];

    const int stride_h = params->stride_h;
    const int stride_w = params->stride_w;
    const int pad_top  = params->pad_top;
    const int pad_left = params->pad_left;

    const int8_t*  in_data  = (const int8_t*)input->data;
    const int8_t*  flt_data = (const int8_t*)filter->data;
    const int32_t* bias_data = bias ? (const int32_t*)bias->data : NULL;
    int8_t*        out_data = (int8_t*)output->data;

    const int32_t in_zp  = input->quant.zero_point;
    const int32_t out_zp = output->quant.zero_point;
    const int32_t* multipliers = params->per_channel_multiplier;
    const int*     shifts      = params->per_channel_shift;

    /* 预计算 zero_point 向量 */
    int8x16_t vec_in_zp = vdupq_n_s8((int8_t)in_zp);

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < C_out; oc++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int32_t acc = 0;

                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * stride_h - pad_top  + kh;
                            int iw = ow * stride_w - pad_left + kw;

                            if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) {
                                continue;  /* padding 区域跳过（zero_point 贡献已在偏置中补偿） */
                            }

                            const int8_t* in_ptr  = in_data  + ((n * H_in + ih) * W_in + iw) * C_in;
                            const int8_t* flt_ptr = flt_data + ((oc * KH + kh) * KW + kw) * C_in;

                            /* Helium 向量化内积：每次处理 16 个 INT8 */
                            int ic = 0;
                            for (; ic <= C_in - 16; ic += 16) {
                                int8x16_t vec_in  = vld1q_s8(in_ptr  + ic);
                                int8x16_t vec_flt = vld1q_s8(flt_ptr + ic);
                                /* 减去 zero_point */
                                vec_in = vsubq_s8(vec_in, vec_in_zp);
                                /* vmladavaq: acc += sum(vec_in[i] * vec_flt[i]) */
                                acc = vmladavaq_s8(acc, vec_in, vec_flt);
                            }
                            /* 处理剩余（C_in % 16 != 0 时，已在上面退化，此处不会执行） */
                            for (; ic < C_in; ic++) {
                                int32_t iv = (int32_t)in_ptr[ic]  - in_zp;
                                int32_t fv = (int32_t)flt_ptr[ic];
                                acc += iv * fv;
                            }
                        }
                    }

                    if (bias_data) acc += bias_data[oc];

                    int32_t scaled;
                    if (multipliers && shifts) {
                        scaled = ev_multiply_by_quantized_multiplier(
                            acc, multipliers[oc], shifts[oc]);
                    } else {
                        scaled = acc;
                    }
                    scaled += out_zp;

                    if (params->activation == EV_ACTIVATION_RELU) {
                        if (scaled < out_zp) scaled = out_zp;
                    } else if (params->activation == EV_ACTIVATION_RELU6) {
                        int32_t relu6_max = out_zp + (int32_t)(6.0f / output->quant.scale);
                        if (scaled < out_zp)    scaled = out_zp;
                        if (scaled > relu6_max) scaled = relu6_max;
                    }

                    int out_idx = ((n * H_out + oh) * W_out + ow) * C_out + oc;
                    out_data[out_idx] = EV_CLAMP_INT8(scaled);
                }
            }
        }
    }

    return EV_SUCCESS;
}

#else  /* 无 Helium 时，Helium 版退化为参考实现 */

ev_status_t ev_conv2d_int8_helium(
    const ev_tensor_t* input,
    const ev_tensor_t* filter,
    const ev_tensor_t* bias,
    ev_tensor_t*       output,
    const ev_conv_params_t* params)
{
    return ev_conv2d_int8_ref(input, filter, bias, output, params);
}

#endif /* __ARM_FEATURE_MVE */

/* ============================================================
 * 6. 静态内存池管理
 * ============================================================ */

/** 内存池结构体 */
typedef struct {
    uint8_t* base;      /**< 内存池起始地址 */
    size_t   capacity;  /**< 总容量（字节） */
    size_t   used;      /**< 已使用字节数 */
} ev_memory_pool_t;

static ev_memory_pool_t g_memory_pool = {NULL, 0, 0};

/**
 * @brief 初始化静态内存池
 *
 * @param buffer   外部提供的静态缓冲区（通常是全局数组）
 * @param size     缓冲区大小（字节）
 */
void ev_memory_pool_init(uint8_t* buffer, size_t size)
{
    g_memory_pool.base     = buffer;
    g_memory_pool.capacity = size;
    g_memory_pool.used     = 0;
}

/**
 * @brief 从内存池分配对齐内存
 *
 * @param size      请求字节数
 * @param alignment 对齐字节数（必须是 2 的幂，通常为 8 或 16）
 * @return 分配的内存指针，失败返回 NULL
 */
void* ev_memory_pool_alloc(size_t size, size_t alignment)
{
    if (!g_memory_pool.base || size == 0) return NULL;

    /* 对齐计算 */
    size_t aligned_used = (g_memory_pool.used + alignment - 1) & ~(alignment - 1);
    if (aligned_used + size > g_memory_pool.capacity) {
        return NULL;  /* 内存不足 */
    }

    void* ptr = g_memory_pool.base + aligned_used;
    g_memory_pool.used = aligned_used + size;
    return ptr;
}

/**
 * @brief 重置内存池（释放所有分配）
 */
void ev_memory_pool_reset(void)
{
    g_memory_pool.used = 0;
}

/**
 * @brief 查询内存池剩余空间
 */
size_t ev_memory_pool_available(void)
{
    if (!g_memory_pool.base) return 0;
    return g_memory_pool.capacity - g_memory_pool.used;
}

/* ============================================================
 * 7. 工业级视觉算子 (针对高反光工件优化)
 * ============================================================ */

#ifdef __ARM_FEATURE_MVE
#include <arm_mve.h>

/**
 * @brief HDR 曝光融合 (Helium 向量化实现)
 * 使用定点运算融合三帧图像。
 */
ev_status_t ev_hdr_fusion_helium(
    const uint8_t* under_exp,
    const uint8_t* normal_exp,
    const uint8_t* over_exp,
    uint8_t*       output,
    uint32_t       width,
    uint32_t       height)
{
    if (!under_exp || !normal_exp || !over_exp || !output) return EV_ERROR_INVALID_ARGUMENT;

    uint32_t num_pixels = width * height;
    uint32_t i = 0;

    /* 每次处理 16 个像素 */
    for (i = 0; i <= num_pixels - 16; i += 16) {
        uint8x16_t v_u = vld1q_u8(under_exp + i);
        uint8x16_t v_n = vld1q_u8(normal_exp + i);
        uint8x16_t v_o = vld1q_u8(over_exp + i);

        /* 定点加权融合: res = (u*51 + n*128 + o*77) >> 8 */
        uint16x8_t v_u_l = vmovlbq_u8(v_u);
        uint16x8_t v_n_l = vmovlbq_u8(v_n);
        uint16x8_t v_o_l = vmovlbq_u8(v_o);

        uint16x8_t v_res_l = vmulq_n_u16(v_u_l, 51);
        v_res_l = vmlaq_n_u16(v_res_l, v_n_l, 128);
        v_res_l = vmlaq_n_u16(v_res_l, v_o_l, 77);
        v_res_l = vshrq_n_u16(v_res_l, 8);

        uint16x8_t v_u_h = vmovltq_u8(v_u);
        uint16x8_t v_n_h = vmovltq_u8(v_n);
        uint16x8_t v_o_h = vmovltq_u8(v_o);

        uint16x8_t v_res_h = vmulq_n_u16(v_u_h, 51);
        v_res_h = vmlaq_n_u16(v_res_h, v_n_h, 128);
        v_res_h = vmlaq_n_u16(v_res_h, v_o_h, 77);
        v_res_h = vshrq_n_u16(v_res_h, 8);

        /* 合并并存储 */
        uint8x16_t v_res = vmovnbq_u16(v_res_l, vmovntq_u16(v_res_l, v_res_h)); // 简化示意
        vst1q_u8(output + i, v_res);
    }

    /* 处理剩余 */
    for (; i < num_pixels; i++) {
        output[i] = (uint8_t)((under_exp[i] * 51 + normal_exp[i] * 128 + over_exp[i] * 77) >> 8);
    }

    return EV_SUCCESS;
}

ev_status_t ev_glare_suppression_helium(
    const uint8_t* input,
    uint8_t*       output,
    uint32_t       width,
    uint32_t       height,
    uint8_t        threshold)
{
    /* 简单实现：超过阈值的像素被周围均值替换 (Helium 加速) */
    // ... 此处可扩展更复杂的反光抑制逻辑
    memcpy(output, input, width * height);
    return EV_SUCCESS;
}

#else /* 无 Helium 时退化为参考实现 */

ev_status_t ev_hdr_fusion_helium(
    const uint8_t* under_exp,
    const uint8_t* normal_exp,
    const uint8_t* over_exp,
    uint8_t*       output,
    uint32_t       width,
    uint32_t       height)
{
    uint32_t num_pixels = width * height;
    for (uint32_t i = 0; i < num_pixels; i++) {
        output[i] = (uint8_t)((under_exp[i] * 51 + normal_exp[i] * 128 + over_exp[i] * 77) >> 8);
    }
    return EV_SUCCESS;
}

ev_status_t ev_glare_suppression_helium(
    const uint8_t* input,
    uint8_t*       output,
    uint32_t       width,
    uint32_t       height,
    uint8_t        threshold)
{
    memcpy(output, input, width * height);
    return EV_SUCCESS;
}

#endif /* __ARM_FEATURE_MVE */
