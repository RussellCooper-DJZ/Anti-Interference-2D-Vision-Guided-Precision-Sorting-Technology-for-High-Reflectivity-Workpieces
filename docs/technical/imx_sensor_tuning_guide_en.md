# Sony IMX Series Sensor Glare Suppression Register Tuning Guide

## 1. Introduction

In industrial machine vision, handling highly reflective metal workpieces (such as stainless steel, electroplated parts, and aluminum alloys) relying solely on backend image processing algorithms is often insufficient to fully recover saturated pixels. This guide provides instructions on how to configure the low-level registers of Sony IMX series sensors (e.g., IMX290, IMX335, IMX415) to suppress specular reflections directly at the **optical acquisition source**.

## 2. Core Registers and Configuration Strategies

### 2.1 Enabling DOL-HDR (Digital Overlap HDR) Mode
Sony's DOL-HDR technology allows the sensor to capture multiple exposures (long, medium, short) sequentially within a single frame period and output interleaved data streams via the CSI-2 interface.
*   **Configuration Method**: Write `0x01` to register `0x300C` (for IMX335) to enable 2-frame DOL-HDR mode.
*   **Purpose**: Short exposure captures edge details in highlight areas, while long exposure captures shadow areas, preventing "all black or all white" issues at the source.

### 2.2 Analog Gain Minimization Strategy
In high-reflectivity environments, blindly increasing gain causes the Full Well Capacity to saturate rapidly.
*   **Configuration Method**: Force the analog gain register (e.g., `0x30E8`) to `0x00` (0dB).
*   **Purpose**: Maximize sensor dynamic range and delay the onset of highlight saturation.

### 2.3 Dynamic Black Level Offset Adjustment
*   **Configuration Method**: Fine-tune register `0x3015`.
*   **Purpose**: Under strong ambient lighting, appropriately raising the black level filters out low-intensity environmental noise, making workpiece edges cleaner.

## 3. Material-Adaptive Tuning Recommendations

| Workpiece Material | Recommended Gain | HDR Mode | Special Register Adjustments |
| :--- | :--- | :--- | :--- |
| **Stainless Steel** | 0 dB | Enabled (DOL-HDR) | Standard Black Level (10-15) |
| **Aluminum Alloy** | 2-3 dB | Enabled | Slightly increase exposure time |
| **Electroplated Parts** | 0 dB | Force Enabled | Ultra-short exposure sequence (100us) |

## 4. Conclusion

By combining low-level register tuning with the RA8P1's built-in Simple ISP, developers can build a closed-loop system of "hardware-level glare suppression + software deep learning inference," completely resolving the challenge of inspecting high-reflectivity workpieces.

---
**Manus AI Industrial Vision Lab**
