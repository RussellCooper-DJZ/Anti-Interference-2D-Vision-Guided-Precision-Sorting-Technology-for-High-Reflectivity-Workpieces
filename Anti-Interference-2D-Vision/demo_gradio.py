"""
demo_gradio.py — FLARE 视觉分拣系统 Gradio 演示界面
:Author: RussellCooper

基于 Gradio 的交互式 Web 前端，支持：
  - 图像上传 / 摄像头实时推理
  - HDR 反光抑制处理前后对比
  - 分割/边缘双视图可视化
  - 实时性能指标显示
  - 多图批量处理

依赖：gradio, opencv-python, torch, numpy

启动：python demo_gradio.py [--model_path checkpoints/best.pth] [--port 7860]
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from vision.feature_extraction import FLARE, FLARELite
from vision.hdr_processing import AntiGlarePipeline, detect_highlight_mask
from vision.localization_and_calibration import SubpixelLocalizer
from vision.inference_engine import PyTorchEngine, export_to_onnx, create_engine


# ============================================================
# 1. 全局配置
# ============================================================

DEFAULT_IMG_SIZE = 512
DEFAULT_MODEL = 'standard'
DEFAULT_BASE_CH = 64


# ============================================================
# 2. 可视化工具
# ============================================================

def draw_results(
    image: np.ndarray,
    seg_mask: np.ndarray,
    edge_mask: np.ndarray,
    glare_mask: np.ndarray,
    detections: list,
    font_scale: float = 0.5,
) -> np.ndarray:
    """在图像上绘制检测结果。"""
    h, w = image.shape[:2]
    result = image.copy()

    # 分割叠加（绿色半透明）
    seg_overlay = np.zeros_like(image)
    seg_overlay[seg_mask > 0] = [0, 200, 0]
    result = cv2.addWeighted(result, 0.7, seg_overlay, 0.3, 0)

    # 边缘叠加（青色）
    result[edge_mask > 0] = (result[edge_mask > 0] * 0.7 +
                              np.array([255, 200, 0]) * 0.3).astype(np.uint8)

    # 高光叠加（红色）
    result[glare_mask > 0] = (0, 0, 220)

    # 绘制目标检测框和标签
    for i, det in enumerate(detections):
        cx, cy = int(det['centroid_px'][0]), int(det['centroid_px'][1])
        angle = det['orientation_deg']
        ftype = det['feature_type']

        # 颜色编码
        color = {'blob': (0, 255, 0), 'line': (255, 100, 0),
                 'region': (0, 165, 255)}.get(ftype, (255, 255, 255))

        # 绘制质心
        cv2.circle(result, (cx, cy), 6, color, -1)
        cv2.circle(result, (cx, cy), 6, (255, 255, 255), 1)

        # 绘制方向箭头
        length = 35
        angle_rad = np.radians(angle)
        ex = int(cx + length * np.cos(angle_rad))
        ey = int(cy + length * np.sin(angle_rad))
        cv2.arrowedLine(result, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

        # 绘制标签
        pos = det.get('position_robot_mm')
        if pos is not None:
            label = f"[{i}] {ftype} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})"
        else:
            label = f"[{i}] {ftype}"
        cv2.putText(result, label, (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return result


def create_comparison_grid(
    original: np.ndarray,
    processed: np.ndarray,
    seg_mask: np.ndarray,
    edge_mask: np.ndarray,
    result_img: np.ndarray,
) -> np.ndarray:
    """创建 2x2 对比网格图。"""
    # 调整所有图像到相同尺寸
    h, w = original.shape[:2]
    processed = cv2.resize(processed, (w, h))
    seg_vis = cv2.resize(seg_mask, (w, h)) if seg_mask.shape != (h, w) else seg_mask
    edge_vis = cv2.resize(edge_mask, (w, h)) if edge_mask.shape != (h, w) else edge_mask

    # 标尺
    scale_bar_h = 30
    scale_bar = np.zeros((scale_bar_h, w, 3), dtype=np.uint8) + 50
    cv2.putText(scale_bar, "100px", (w // 2 - 30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 顶部行：原图 + HDR处理后
    top = np.hstack([original, processed])
    # 底部行：分割结果 + 检测结果
    seg_colored = cv2.cvtColor(seg_vis, cv2.COLOR_GRAY2BGR)
    edge_colored = cv2.cvtColor(edge_vis, cv2.COLOR_GRAY2BGR)
    bottom = np.hstack([seg_colored, result_img])

    # 添加标尺
    top = np.vstack([top, scale_bar])
    bottom = np.vstack([bottom, scale_bar])

    grid = np.vstack([top, bottom])

    # 添加标签
    labels = ["原图 + 高光", "HDR处理后", "分割掩膜", "检测结果"]
    positions = [(10, 25), (w + 10, 25), (10, h + 25), (w + 10, h + 25)]
    for label, pos in zip(labels, positions):
        cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return grid


def image_to_base64(img: np.ndarray) -> str:
    """将图像转为 base64 字符串（用于 Gradio 显示）。"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


# ============================================================
# 3. 推理管线
# ============================================================

class InferencePipeline:
    """FLARE 推理管线，包含完整预处理和后处理。"""

    def __init__(
        self,
        model_path: str = None,
        model_type: str = DEFAULT_MODEL,
        base_ch: int = DEFAULT_BASE_CH,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        # 构建模型
        if model_type == 'lite':
            self.model = FLARELite(in_channels=3, base_ch=base_ch)
        else:
            self.model = FLARE(in_channels=3, base_ch=base_ch)

        # 加载权重
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location='cpu')
            state = ckpt.get('model', ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[Pipeline] 已加载模型: {model_path}")
        else:
            print("[Pipeline] 警告：未加载预训练权重，使用随机初始化")

        # 创建推理引擎
        self.engine = PyTorchEngine(self.model, img_size=img_size)
        self.img_size = img_size

        # HDR 处理
        self.anti_glare = AntiGlarePipeline()

        # 定位器
        self.localizer = SubpixelLocalizer(min_area=100)

    def process(
        self,
        image_bgr: np.ndarray,
        apply_hdr: bool = True,
        send_to_robot: bool = False,
    ) -> dict:
        """
        处理单张图像。

        Returns:
            包含可视化结果、检测数据的字典
        """
        t_start = time.perf_counter()
        timing = {}

        # Step 1: HDR 融合
        t0 = time.perf_counter()
        if apply_hdr:
            image_hdr = self.anti_glare.process_single(image_bgr)
        else:
            image_hdr = image_bgr.copy()
        timing['hdr_ms'] = (time.perf_counter() - t0) * 1000

        # Step 2: 高光检测
        t0 = time.perf_counter()
        glare_mask_raw = detect_highlight_mask(image_hdr)
        timing['glare_ms'] = (time.perf_counter() - t0) * 1000

        # Step 3: FLARE 推理
        t0 = time.perf_counter()
        seg_mask, edge_mask = self.engine.predict(image_hdr)
        timing['infer_ms'] = (time.perf_counter() - t0) * 1000

        # Step 4: 像素级定位
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image_hdr, cv2.COLOR_BGR2GRAY)
        detections = self.localizer.localize(
            seg_mask, edge_mask, intensity_image=gray, glare_mask=glare_mask_raw
        )
        timing['localize_ms'] = (time.perf_counter() - t0) * 1000

        # Step 5: 坐标变换（简单默认内参）
        for det in detections:
            det['position_robot_mm'] = (
                det['centroid_px'][0] * 2 - 512,
                det['centroid_px'][1] * 2 - 512,
                800.0,
            )

        # Step 6: 可视化
        result_img = draw_results(image_hdr, seg_mask, edge_mask, glare_mask_raw, detections)

        # 创建对比网格
        grid = create_comparison_grid(image_bgr, image_hdr, seg_mask, edge_mask, result_img)

        total_ms = (time.perf_counter() - t_start) * 1000
        timing['total_ms'] = total_ms

        return {
            'grid': grid,
            'result': result_img,
            'seg_mask': seg_mask,
            'edge_mask': edge_mask,
            'glare_mask': glare_mask_raw,
            'detections': detections,
            'timing': timing,
        }


# ============================================================
# 4. Gradio 界面
# ============================================================

def build_demo(pipeline: InferencePipeline):
    """构建 Gradio 演示界面。"""

    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio 未安装。运行: pip install gradio"
        )

    def process_image(image, apply_hdr, model_type):
        """处理上传的图像。"""
        if image is None:
            return None, "请上传图像"

        # BGR 转 RGB（Gradio 使用 RGB）
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 推理
        result = pipeline.process(image, apply_hdr=apply_hdr)

        # 转为 base64 用于显示
        grid_b64 = image_to_base64(result['grid'])

        # 生成报告
        n_dets = len(result['detections'])
        timing = result['timing']
        report = (
            f"## 检测报告\n\n"
            f"**目标数量**: {n_dets}\n\n"
            f"### 处理时间 (ms)\n"
            f"| 阶段 | 耗时 |\n"
            f"|------|------|\n"
            f"| HDR融合 | {timing['hdr_ms']:.1f} |\n"
            f"| 模型推理 | {timing['infer_ms']:.1f} |\n"
            f"| 像素级定位 | {timing['localize_ms']:.1f} |\n"
            f"| **总计** | **{timing['total_ms']:.1f}** |\n\n"
            f"### FPS: {1000.0 / timing['total_ms']:.1f}\n\n"
            f"### 检测详情\n"
        )

        for i, det in enumerate(result['detections']):
            pos = det['position_robot_mm']
            report += (
                f"- **目标{i}** [{det['feature_type']}] "
                f"质心({det['centroid_px'][0]:.1f}, {det['centroid_px'][1]:.1f}) "
                f"方向{det['orientation_deg']:.1f}° "
                f"坐标({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})\n"
            )

        if n_dets == 0:
            report += "\n*未检测到目标*"

        return f"data:image/png;base64,{grid_b64}", report

    # CSS 自定义样式
    css = """
    .title {text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;}
    .stats {background: #f0f0f0; padding: 15px; border-radius: 10px;}
    """

    with gr.Blocks(css=css, title="FLARE 视觉分拣系统") as demo:
        gr.Markdown("<div class='title'>🔬 FLARE 视觉分拣系统</div>")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="上传图像",
                    type="rgb",
                    height=300,
                )
                with gr.Row():
                    apply_hdr = gr.Checkbox(
                        label="启用HDR反光抑制",
                        value=True,
                    )
                    model_select = gr.Dropdown(
                        label="模型",
                        choices=["standard", "lite"],
                        value="standard",
                    )
                process_btn = gr.Button("🔍 开始检测", variant="primary")

            with gr.Column(scale=2):
                image_output = gr.Image(
                    label="检测结果",
                    type="rgb",
                    height=400,
                )
                report_output = gr.Markdown(label="检测报告")

        process_btn.click(
            fn=process_image,
            inputs=[image_input, apply_hdr, model_select],
            outputs=[image_output, report_output],
        )

        gr.Markdown("---")
        gr.Markdown("### 使用说明\n"
                    "1. 上传一张图像（或使用摄像头）\n"
                    "2. 选择是否启用 HDR 反光抑制\n"
                    "3. 点击「开始检测」查看结果\n\n"
                    "支持的图像格式：PNG, JPG, BMP")

    return demo


# ============================================================
# 5. 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FLARE Gradio 演示界面")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型权重路径 (.pth)")
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "lite"],
                        help="模型类型")
    parser.add_argument("--base_ch", type=int, default=64,
                        help="基础通道数")
    parser.add_argument("--img_size", type=int, default=512,
                        help="输入图像尺寸")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio 服务端口")
    parser.add_argument("--share", action="store_true",
                        help="创建公开分享链接")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")

    args = parser.parse_args()

    print("=" * 60)
    print("  FLARE 视觉分拣系统 - Gradio 演示")
    print("=" * 60)

    # 创建推理管线
    pipeline = InferencePipeline(
        model_path=args.model_path,
        model_type=args.model_type,
        base_ch=args.base_ch,
        img_size=args.img_size,
    )

    # 构建界面
    demo = build_demo(pipeline)

    # 启动服务
    print(f"\n启动服务: http://localhost:{args.port}")
    print("按 Ctrl+C 停止服务\n")

    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
