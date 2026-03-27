# Anti-Interference 2D Vision-Guided Precision Sorting Technology for High-Reflectivity Workpieces

## 1. Introduction

This technical solution aims to address the core challenges of "unclear imaging, inaccurate positioning, and unstable gripping" faced in 2D vision-guided sorting of high-reflectivity metal workpieces. By integrating advanced image processing, deep learning, and robotic control technologies, it achieves high-precision, high-stability recognition and positioning of high-reflectivity workpieces to meet the stringent production requirements of industrial sites. Currently, visual inspection of high-reflectivity surfaces is a significant challenge in machine vision, and researchers worldwide are actively exploring the combination of multimodal imaging, deep learning, and other cutting-edge technologies to overcome the limitations of traditional methods [10, 11].

## 2. Core Technical Indicators Review

According to user requirements, this solution needs to meet the following core technical indicators:

| Indicator Category | Specific Requirements |
| :----------------- | :-------------------- |
| **Recognition Accuracy & Stability** | Workpiece recognition success rate ≥ 99.5% (under typical industrial lighting and strong interference lighting conditions); False detection rate ≤ 0.1% |
| **Positioning Accuracy & Speed** | 2D planar positioning accuracy: error ≤ ±0.2mm (or ≤ 0.5 pixels); Angular positioning accuracy: error ≤ ±0.5°; Vision processing cycle: ≤ 300ms (from image capture to coordinate output) |
| **Algorithm Robustness** | Adapt to a certain degree of oil stains and fingerprint interference on the workpiece surface; Recognition rate without significant degradation under ±20% light source brightness fluctuation; Support general recognition of at least 3 different materials (e.g., stainless steel, aluminum alloy, electroplated parts) of high-reflectivity workpieces |

## 3. Optical Solution Design

For the imaging challenges of high-reflectivity workpieces, the design of the optical solution is crucial. Its goal is to suppress specular reflection and ambient light interference as much as possible, while highlighting the true features of the workpiece. Domestic and international research indicates that a reasonable optical design is fundamental to the success of visual inspection of high-reflectivity surfaces [10, 11].

### 3.1 Light Source Selection and Layout

Considering the high-reflectivity characteristics, traditional direct light sources tend to cause local overexposure. This solution will adopt the following strategies:

*   **Diffuse Light Source**: Prioritize the use of integrating sphere light sources, dome light sources, or ring diffuse light sources to provide uniform, shadow-free illumination, effectively reducing local highlight areas caused by specular reflection. For materials like stainless steel and aluminum alloy, diffuse light can better reveal their contours.
*   **Polarized Light Illumination**: For electroplated parts with particularly severe specular reflection, consider using polarized light sources with polarized cameras or polarized filters. By adjusting the polarization direction, specular reflection can be effectively filtered out, thereby revealing the true texture and edges of the workpiece surface. Research shows that polarized imaging has significant advantages in suppressing specular reflection and enhancing surface details [12, 13]. This will be a key focus for software-level simulation and optimization in this solution.
*   **Multi-angle/Multi-light Source Combination**: On some complex-shaped workpieces, a single light source may not provide complete coverage. A combination of low-power light sources at multiple angles can be used to obtain more comprehensive information through image fusion technology.

### 3.2 Camera and Lens Selection

*   **Camera**: Select an industrial camera with high resolution (e.g., 5 megapixels or more) and high frame rate to meet the requirements for positioning accuracy and vision processing cycle. Additionally, the camera needs to have a good signal-to-noise ratio and Wide Dynamic Range (HDR) function to cope with lighting variations. High Dynamic Range imaging technology performs excellently in handling overexposure and underexposure issues in high-reflectivity scenes [14].
*   **Lens**: Choose a low-distortion, high-resolution fixed-focal-length lens to ensure the geometric accuracy of the image. Calculate the appropriate focal length and working distance based on the workpiece size and field of view.

### 3.3 Image Acquisition Strategy

*   **Multi-exposure Acquisition**: To achieve High Dynamic Range (HDR) imaging, the camera will perform multi-exposure acquisition, i.e., capturing multiple images at different exposure times (underexposed, normally exposed, overexposed). These images will be fused in subsequent algorithms. This method has been widely applied in 3D measurement and feature extraction of high-reflectivity surfaces [10, 14].
*   **Polarized Image Acquisition (Simulation)**: Without increasing hardware costs, this solution will investigate simulating the de-reflection effect of polarized light imaging through software algorithms, for example, by analyzing the reflection characteristics in multi-exposure images to suppress specular reflection. While hardware polarized cameras offer better results, software simulation can serve as a low-cost alternative, utilizing image processing techniques (such as reflection separation based on color or texture) to suppress specular reflection [12].

## 4. Algorithm Principles and Architecture

This solution's algorithm architecture will be divided into three main modules: image pre-processing, feature extraction and positioning, and robot guidance and control. The rapid development of deep learning provides powerful tools for solving complex vision tasks, especially in feature extraction and robustness [15, 16].

### 4.1 Image Pre-processing Module

#### 4.1.1 High Dynamic Range (HDR) Imaging and Multi-exposure Fusion

To address local overexposure and "blackout" phenomena caused by high reflectivity, multi-exposure fusion technology is employed. By acquiring multiple images at different exposure times (underexposed, normally exposed, overexposed) and using fusion algorithms (such as Debevec [1], Mertens [2], or Drago [3] algorithms), an HDR image with a wide dynamic range is generated. This image can simultaneously display details in both highlight and shadow areas, effectively restoring the true features of the workpiece surface.

*   **Principle**: Map pixel values of different exposure images to a unified radiance space, and then generate a more information-rich image through weighted averaging or gradient domain fusion methods.
*   **Advantages**: Effectively solves the dynamic range limitations of traditional single images, providing high-quality input for subsequent feature extraction.

#### 4.1.2 Polarized Light Imaging Principle Simulation and Specular Reflection Suppression

Although hardware polarized cameras are costly, this solution will explore simulating the de-reflection effect of polarized light imaging through software algorithms. This may involve analyzing the reflection characteristics in multi-exposure images, or utilizing image processing techniques (such as reflection separation based on color or texture) to suppress specular reflection. This method aims to achieve similar effects without using physical polarizing filters, enhancing the visibility of edges and textures of high-reflectivity workpieces [12].

*   **Principle**: Specular reflection light has polarization characteristics, while diffuse reflection light does not. By analyzing the intensity of reflected light in different directions in the image, attempts can be made to separate the specular reflection component. Software simulation will try to achieve similar effects without using physical polarizing filters.
*   **Advantages**: Improves the visibility of edges and textures of high-reflectivity workpieces without increasing hardware costs.

#### 4.1.3 Adaptive Image Enhancement

After HDR fusion and reflection suppression, adaptive image enhancement is performed, including contrast stretching, brightness adjustment, and noise filtering. The goal is to further suppress light spot noise and improve overall image quality without sacrificing edge accuracy.

*   **Methods**: CLAHE (Contrast Limited Adaptive Histogram Equalization), Non-local Means filtering, or Guided Filter, etc.

### 4.2 Robust Feature Extraction and Positioning Algorithm Against Lighting Interference

This module is key to achieving high-precision positioning and will employ deep learning methods to overcome the limitations of traditional algorithms under complex lighting conditions. Deep learning has demonstrated powerful capabilities in computational imaging and automated optical inspection [15, 16].

#### 4.2.1 Deep Learning-Based Feature Extraction Network

Convolutional Neural Networks (CNNs) or Transformer-based networks are used for workpiece feature extraction. This network will directly learn robust features of the workpiece from pre-processed images, rather than relying on traditional gradient-based methods. Research shows that deep learning models can effectively handle complex lighting and surface variations [15, 16].

*   **Network Architecture**: U-Net [6], Mask R-CNN [7], or YOLO [8] object detection and segmentation networks can be considered and customized according to actual needs. For contour extraction, semantic segmentation networks (such as DeepLabV3+) may be more suitable.
*   **Training Data**: Requires a large number of annotated images with different lighting, materials, and surface defects (oil stains, fingerprints) for training. Data augmentation techniques (such as random brightness, contrast, noise, rotation, scaling) will be used to improve the model's generalization ability.
*   **Robustness**: Features learned by deep learning models can effectively distinguish between true workpiece boundaries and "false edges" formed by reflections, thereby achieving precise contour extraction under complex lighting.

#### 4.2.2 False Edge Removal Technology

Based on deep learning feature extraction, post-processing algorithms are combined to further optimize edge detection results.

*   **Methods**: Utilize morphological operations, connected component analysis, geometric constraints (e.g., known workpiece shape priors), etc., to refine the edges output by the network, removing "false edges" or noise points that do not conform to the true workpiece features.
*   **Advantages**: Ensures that the extracted edges are the true physical boundaries of the workpiece, not lighting artifacts.

#### 4.2.3 Sub-pixel Level Positioning

To achieve a positioning accuracy of ±0.2mm, sub-pixel level edge detection and fitting are required on the extracted workpiece contours. Domestic and international research has made significant progress in sub-pixel positioning, especially in high-resolution image processing [10, 11].

*   **Methods**: Based on Zernike moments, Gaussian fitting, or interpolation methods, the rough pixel-level edges are refined to obtain sub-pixel accurate edge points. Then, the workpiece's geometric center, principal axis direction, and other key positioning information are fitted using least squares or other optimization algorithms.
*   **Advantages**: Significantly improves positioning accuracy, meeting stringent industrial requirements.

### 4.3 High-Precision Hand-Eye Calibration and Visual Servo Control

#### 4.3.1 Hand-Eye Calibration Model

Establish a high-precision hand-eye calibration model to achieve precise mapping between the image pixel coordinate system and the robot base coordinate system. This will employ classic Tsai-Lenz [4] or Park-Martin [5] methods. In recent years, online hand-eye calibration and deep learning-based calibration methods have also become research hotspots, improving calibration flexibility and robustness [17, 18, 19].

*   **Principle**: By capturing images of a calibration board at different poses, the transformation relationship between the camera coordinate system and the calibration board coordinate system is obtained, as well as the transformation relationship between the robot end-effector coordinate system and the robot base coordinate system. Then, through mathematical derivation, the transformation matrix (hand-eye matrix) between the camera coordinate system and the robot end-effector coordinate system is solved.
*   **Accuracy Improvement**: High-precision calibration boards, multiple repeated measurements, optimization algorithms (such as Levenberg-Marquardt), and error compensation techniques are used to ensure calibration accuracy.

#### 4.3.2 Visual Servo Control

In cases where visual positioning coordinates exhibit slight fluctuations or during robot movement, a robot motion compensation strategy based on visual feedback is adopted to achieve precise gripping of workpieces.

*   **Methods**: Position-Based Visual Servo (PBVS) or Image-Based Visual Servo (IBVS). PBVS calculates the target's pose in the camera coordinate system, then converts it to the robot base coordinate system to drive robot movement. IBVS directly uses image feature errors to control robot movement.
*   **Robustness**: Combine predictive control, Kalman filtering, and other techniques to smooth visual positioning results, reduce robot jitter, and improve gripping stability.

## 5. Conclusion and Outlook

This solution, by combining advanced optical design, image pre-processing, deep learning feature extraction, and high-precision robot control technologies, aims to provide a comprehensive, robust, and cost-effective solution for 2D vision-guided precision sorting of high-reflectivity workpieces. Future work will include actual algorithm deployment, performance optimization, and generalization capabilities for more complex workpieces.

## 6. References

[1] Debevec, P. E., & Malik, J. (1997). Recovering high dynamic range radiance maps from photographs. *Proceedings of the 24th annual conference on Computer graphics and interactive techniques* (pp. 369-378). ACM.
[2] Mertens, T., Kautz, J., & Van Reeth, F. (2009). Exposure fusion: A simple and effective way to combine pictures with different exposures. *Computer Graphics Forum, 28*(1), 161-171.
[3] Drago, F., Myszkowski, K., Annen, T., & Seidel, H. P. (2003). Adaptive logarithmic mapping for displaying high contrast scenes. *Computer Graphics Forum, 22*(3), 419-426.
[4] Tsai, R. Y., & Lenz, R. K. (1989). A new technique for fully autonomous and efficient 3D robotics hand/eye calibration. *IEEE Transactions on Robotics and Automation, 5*(3), 345-358.
[5] Park, F. C., & Martin, B. J. (1994). Robot sensor calibration: A review. *Robotica, 12*(6), 505-518.
[6] Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 3431-3440).
[7] He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *Proceedings of the IEEE international conference on computer vision* (pp. 2961-2969).
[8] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 779-788).
[9] Su, S. (2022). Research on the Hand–Eye Calibration Method of Variable ... *Sensors, 12*(9), 4415.
[10] "Progress in in-situ detection technology and application of high dynamic range structured light stripes." (2025). *Acta Optica Sinica*.
[11] "Review of automatic optical (vision) inspection technology and its application in defect detection." (2025). *Acta Optica Sinica*.
[12] "Research status of deep learning polarized image fusion." (2025). *Infrared and Laser Engineering*.
[13] Instrumentation for Estimating Surface Radiometry. (n.d.). *DTU Orbit*.
[14] Debevec, P. E., & Malik, J. (1997). Recovering high dynamic range radiance maps from photographs. *Proceedings of the 24th annual conference on Computer graphics and interactive techniques* (pp. 369-378). ACM.
[15] Abu Ebayyeh, A. A. R. M. (2022). *Deep Learning for Automatic Optical Inspection and Quality ...*. Brunel University London.
[16] Wang, J. (n.d.). *Frontier Progress in Computational Imaging*. Carnegie Mellon University.
[17] Lin, W. (2022). Research of Online Hand–Eye Calibration Method Based ... *Sensors, 12*(9), 4415.
[18] Bahadir, O. (2023). A Deep Learning-Based Hand-eye Calibration Approach ... *University of Glasgow*.
[19] Li, L. (2023). Automatic Robot Hand-Eye Calibration Enabled by ... *arXiv preprint arXiv:2311.01335*.
