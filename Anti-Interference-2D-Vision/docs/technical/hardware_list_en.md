# Hardware List

The following hardware components are required for the high-reflectivity workpiece anti-interference 2D vision-guided precision sorting solution:

## 1. Core Processing Unit

*   **Renesas RA8P1 Microcontroller Development Board**: Equipped with an Arm Cortex-M85 core, used for running embedded vision algorithms and control logic. It is recommended to use an official evaluation board or a custom development board.

## 2. Image Acquisition System

*   **Industrial Camera**:
    *   **Type**: High-resolution (e.g., 5 megapixels or higher), high-frame-rate (e.g., 60fps or higher) global shutter CMOS industrial camera.
    *   **Interface**: CSI-2 interface compatible with RA8P1 or other high-speed interfaces (e.g., USB3.0, requiring a bridge chip).
    *   **Functionality**: Equipped with HDR capabilities or supporting multi-exposure modes.
*   **Industrial Lens**:
    *   **Type**: Low-distortion, high-resolution fixed-focal-length lens.
    *   **Focal Length**: Select an appropriate focal length based on the working distance and field of view.
*   **Polarizing Filter (Optional)**: If hardware polarizing imaging is used, a polarizing filter needs to be installed in front of the lens.

## 3. Lighting System

*   **Diffuse Light Source**:
    *   **Type**: Integrating sphere light, dome light, or ring diffuse light source.
    *   **Power**: Select appropriate power based on workpiece reflectivity and ambient light conditions to ensure uniform illumination.
*   **Polarizing Light Source (Optional)**: If hardware polarizing imaging is used, a polarizing light source is required.
*   **Light Source Controller**: Used for precise control of light source brightness, flash mode, etc.

## 4. Robotic System

*   **Industrial Robot**: Six-axis or multi-axis industrial robot with high-precision repeatability.
*   **Robot Controller**: Communicates with RA8P1, receives vision guidance commands, and executes gripping actions.
*   **End Effector**: Select an appropriate gripper or suction cup based on workpiece shape and material.

## 5. Calibration Tools

*   **High-Precision Calibration Board**: Used for camera intrinsic calibration and hand-eye calibration, such as a chessboard, ChArUco board, or dot array calibration board.

## 6. Auxiliary Equipment

*   **Power Supply**: Provides stable power to all hardware components.
*   **Industrial PC**: Used for algorithm development, model training, hand-eye calibration, and communication debugging with RA8P1.
*   **Monitor**: For debugging and monitoring.
*   **Connection Cables**: Including camera data cables, power cables, communication cables, etc.

## 7. Software Environment

*   **Development Toolchain**: Renesas e2 studio IDE, Arm GNU Toolchain.
*   **FSP (Flexible Software Package)**: Renesas-provided software development package, including drivers and middleware.
*   **TensorFlow Lite for Microcontrollers (TFLM)**: Used for deploying deep learning models on RA8P1.
*   **OpenCV (PC Version)**: Used for PC-side image processing algorithm development and verification.
*   **Python/PyTorch/TensorFlow (PC Version)**: Used for deep learning model training and conversion.
*   **Robot Programming Environment**: Programming software provided by the robot manufacturer.

## 8. Enhanced Components (2026-04-23 Update)

### 8.1 PBR Lighting System Components

For the enhanced PBR (Physically Based Rendering) lighting simulation:

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Multi-directional LED Array** | ≥4 directions, 0-100% intensity control | Simulate complex industrial lighting |
| **High-intensity Directional LED** | ≥10000 Lux equivalent | `pbr_sun` mode sunlight simulation |
| **Reflective Surface Setup** | Matte black + mirror surfaces | `pbr_mixed` water reflection mode |

### 8.2 Inference Computing Components

| Component | Specification | Use Case |
|-----------|---------------|----------|
| **NVIDIA GPU (Training)** | CUDA-capable GPU, ≥8GB VRAM | PBR data generation, model training |
| **ONNX Runtime (Edge)** | Helium-optimized build for RA8P1 | Embedded inference, ~40ms latency |
| **TensorRT (Optional)** | NVIDIA GPU required | PC/Edge inference, ~10ms latency |

### 8.3 ABB Robot Interface Components

For the AbbRobotEGM UDP EGM protocol interface:

| Component | Specification | Purpose |
|-----------|---------------|----------|
| **ABB Robot Controller** | IRC5 or newer with EGM option | Real robot control |
| **Network Switch** | 1000Mbps, low latency | UDP EGM communication (port 6510) |
| **EGM Configuration** | Protobuf or binary fallback | RobotStudio integration |
