# 高反光工件抗干扰2D视觉引导精准分拣技术方案
# Anti-Interference 2D Vision-Guided Precision Sorting Technology for High-Reflectivity Workpieces
# Störungsfreie 2D-Vision-geführte Präzisionssortiertechnologie für hochreflektierende Werkstücke

## 简介 (Chinese Introduction)

本项目旨在开发一套针对高反光金属工件的抗干扰2D视觉引导精准分拣技术。通过集成先进的光学设计、图像预处理、深度学习特征提取和高精度机器人控制，解决工业生产中高反光工件“看不清、定不准、抓不稳”的核心难题。本方案不仅提供了基于PC端的算法实现，还特别针对瑞萨RA8P1嵌入式平台进行了优化和代码示例，以实现低成本、高性能的边缘视觉解决方案。

## Introduction (English Introduction)

This project aims to develop an anti-interference 2D vision-guided precision sorting technology for high-reflectivity metal workpieces. By integrating advanced optical design, image pre-processing, deep learning feature extraction, and high-precision robotic control, it addresses the core challenges of "unclear imaging, inaccurate positioning, and unstable gripping" encountered with high-reflectivity workpieces in industrial production. This solution provides not only PC-based algorithm implementations but also optimized code examples specifically for the Renesas RA8P1 embedded platform, aiming for a low-cost, high-performance edge vision solution.

## Einführung (German Introduction)

Dieses Projekt zielt darauf ab, eine störungsfreie 2D-Vision-geführte Präzisionssortiertechnologie für hochreflektierende Metallwerkstücke zu entwickeln. Durch die Integration fortschrittlicher optischer Designs, Bildvorverarbeitung, Deep-Learning-Merkmalsextraktion und hochpräziser Robotersteuerung werden die Kernherausforderungen der "unklaren Bildgebung, ungenauen Positionierung und instabilen Greifens" bei hochreflektierenden Werkstücken in der industriellen Produktion gelöst. Diese Lösung bietet nicht nur PC-basierte Algorithmusimplementierungen, sondern auch optimierte Codebeispiele speziell für die eingebettete Renesas RA8P1-Plattform, um eine kostengünstige, leistungsstarke Edge-Vision-Lösung zu realisieren.

## 目录 (Table of Contents)

*   [详细技术方案 (Detailed Technical Solution - Chinese)](./high_reflectivity_sorting_solution.md)
*   [Detailed Technical Solution (English)](./high_reflectivity_sorting_solution_en.md)
*   [Detaillierte technische Lösung (Deutsch)](./high_reflectivity_sorting_solution_de.md)
*   [瑞萨 RA8P1 嵌入式视觉方案 (Renesas RA8P1 Embedded Vision Solution - Chinese)](./ra8p1_embedded_vision_solution.md)
*   [Renesas RA8P1 Embedded Vision Solution (English)](./ra8p1_embedded_vision_solution_en.md)
*   [Renesas RA8P1 Embedded Vision Lösung (Deutsch)](./ra8p1_embedded_vision_solution_de.md)
*   [硬件清单 (Hardware List - Chinese)](./hardware_list_zh.md)
*   [Hardware List (English)](./hardware_list_en.md)
*   [Hardwareliste (Deutsch)](./hardware_list_de.md)
*   [代码文件 (Code Files)](#代码文件)

## 代码文件 (Code Files)

### PC 端算法实现 (PC-side Algorithm Implementation)

*   `hdr_processing.py`: HDR 图像处理模块 (HDR Image Processing Module)
*   `feature_extraction.py`: 深度学习特征提取模块 (Deep Learning Feature Extraction Module)
*   `localization_and_calibration.py`: 亚像素定位与手眼标定模块 (Sub-pixel Localization and Hand-Eye Calibration Module)
*   `main_system.py`: 完整系统集成与性能测试示例 (Full System Integration and Performance Test Example)

### 瑞萨 RA8P1 嵌入式端算法实现 (Renesas RA8P1 Embedded Algorithm Implementation)

*   `ra8p1_tflm_adapter.py`: TFLite Micro 模型转换脚本 (TFLite Micro Model Conversion Script)
*   `ra8p1_helium_processing.c`: 基于 Helium 指令集的图像预处理 C 代码示例 (Helium Instruction Set-based Image Pre-processing C Code Example)
*   `ra8p1_main_app.c`: 基于瑞萨 FSP 的系统集成框架 (Renesas FSP-based System Integration Framework)

## 如何使用 (How to Use / Wie zu verwenden)

请参考各文档和代码文件中的详细说明。

Please refer to the detailed instructions in each document and code file.

Bitte beachten Sie die detaillierten Anweisungen in jedem Dokument und jeder Codedatei.
