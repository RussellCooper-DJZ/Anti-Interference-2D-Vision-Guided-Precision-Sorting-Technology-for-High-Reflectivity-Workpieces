# Hardware and Optical Design for Anti-Interference 2D Vision-Guided Sorting of High-Reflectivity Workpieces

## 1. Solution Overview

This solution addresses the precise sorting requirements of high-reflectivity metal workpieces (e.g., stainless steel, aluminum alloy, electroplated parts) in industrial environments. It fully leverages the built-in **Simple ISP** functionality of the Renesas **RA8P1** to achieve high-performance vision recognition using low-cost hardware combinations. The core idea is to process Bayer RAW data from low-cost CMOS sensors using the Simple ISP, combined with multi-exposure HDR technology and polarized optical design to eliminate specular reflection interference.

## 2. Hardware Selection List

| Component Category | Recommended Model/Specification | Selection Reason |
| :--- | :--- | :--- |
| **Core Controller** | **Renesas RA8P1** (Cortex-M85) | Built-in **Simple ISP**, supports RAW8/10/12 input; Helium acceleration instruction set boosts AI inference speed; low power consumption, high integration. |
| **Image Sensor** | **OmniVision OV5640** or **Sony IMX Series** (RAW output) | Supports Bayer RGB RAW output, perfectly matching Simple ISP; 5MP resolution meets precision requirements; extremely low cost. |
| **Lens** | 12mm/16mm Industrial Fixed-Focus Lens (Low Distortion) | Selected based on working distance to ensure the workpiece occupies sufficient pixels in the field of view; low distortion favors sub-pixel positioning. |
| **Polarizer** | Linear Polarizer (Mounted in front of the lens) | Works with polarized light sources to eliminate over 90% of specular reflection light through the orthogonal polarization principle. |
| **Light Source** | **Polarized Bar Light** or **Polarized Ring Light** | 500-1000 Lux brightness; built-in polarizing film reduces ambient light interference and suppresses surface glare on the workpiece. |
| **Robot Interface** | UART / CAN / Ethernet | RA8P1's rich interfaces can communicate directly with mainstream industrial robots (e.g., KUKA, FANUC). |

## 3. Optical Design

### 3.1 Polarized Imaging Principle
Specular reflection light from the surface of high-reflectivity workpieces has strong polarization characteristics. By installing a polarizer in front of the light source and another orthogonal (rotated 90°) polarizer in front of the lens, strong directly reflected light can be effectively filtered out, leaving only diffuse reflection light that carries workpiece surface texture information.

### 3.2 Multi-exposure HDR Strategy
Utilize the RA8P1 to control the sensor for rapid triple exposure (under-exposure, normal, over-exposure):
1.  **Under-exposure**: Captures edge details in high-reflectivity areas, preventing feature loss due to over-exposure.
2.  **Normal Exposure**: Obtains clear images of the background and normal areas.
3.  **Over-exposure**: Extracts features from dark areas (e.g., areas with oil stains or fingerprint interference).
After pre-processing by the Simple ISP, software algorithms perform fusion to generate high dynamic range images.

## 4. Simple ISP Parameter Tuning Strategy

For high-reflectivity scenes, it is recommended to set the following parameters via the V4L2 interface in the RA8P1:

| Parameter ID | Recommended Setting | Purpose |
| :--- | :--- | :--- |
| `V4L2_CID_RZ_ISP_GAMMA` | 150 - 200 (1.5 - 2.0) | Enhances contrast in dark areas and suppresses highlights. |
| `V4L2_CID_RZ_ISP_2DNR` | 70 - 100 | Eliminates planar noise from the sensor at high gain. |
| `V4L2_CID_RZ_ISP_EMP` | 2 (Normal) | Strengthens workpiece edges and improves sub-pixel positioning accuracy. |
| `V4L2_CID_RZ_ISP_BL` | 10 - 20 | Appropriately increases the black level to filter out weak background reflections. |

## 5. Cost Advantage Analysis
*   **No External ISP Required**: Saves approximately $5-$10 in hardware costs.
*   **Low-Cost Sensors**: Supports common Bayer RAW sensors available on the market, reducing sensor costs by over 30% compared to sensors with integrated ISPs.
*   **High Integration**: The RA8P1 single chip completes image acquisition, ISP processing, AI inference, and motion control, simplifying circuit design.
