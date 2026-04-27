"""
demo_streamlit.py — FLARE 视觉分拣系统 Streamlit 演示界面
:Author: RussellCooper

基于 Streamlit 的交互式 Web 前端，支持：
  - 图像上传 / 文件夹批量处理
  - HDR 反光抑制处理前后对比滑块
  - 分割/边缘双视图可视化
  - 实时性能指标仪表盘
  - 视频文件逐帧处理
  - 检测结果导出 (CSV/JSON)

依赖：streamlit, opencv-python, torch, numpy, pandas

启动：streamlit run demo_streamlit.py [--browser.server_port 8501]
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from vision.feature_extraction import FLARE, FLARELite
from vision.hdr_processing import AntiGlarePipeline, detect_highlight_mask
from vision.localization_and_calibration import SubpixelLocalizer
from vision.inference_engine import PyTorchEngine, create_engine
from vision.gripper_simulation import draw_grasp_on_image, draw_all_grasps


# ============================================================
# Streamlit 配置
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="FLARE 视觉分拣系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. 全局配置
# ============================================================

DEFAULT_IMG_SIZE = 512


# ============================================================
# 2. 推理管线
# ============================================================

@st.cache_resource
def load_pipeline(model_path: str = None, model_type: str = 'standard',
                  base_ch: int = 32, compute_gripper: bool = True,
                  gripper_width: int = 40):
    """加载推理管线（使用缓存）。"""
    import torch

    # 自动选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 如果提供了模型路径，先读取checkpoint配置
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        ckpt_args = ckpt.get('args', {})
        # 从checkpoint自动获取base_ch和model_type
        checkpoint_base_ch = ckpt_args.get('base_ch', 32)
        checkpoint_model = ckpt_args.get('model', 'standard')

        # 使用checkpoint的配置（如果用户没有显式指定）
        if base_ch == 64:  # 默认值，说明用户没有指定
            base_ch = checkpoint_base_ch
        if model_type == 'standard':  # 默认值
            model_type = checkpoint_model

        st.sidebar.success(f"Checkpoint配置: base_ch={checkpoint_base_ch}, model={checkpoint_model}")

    # 根据配置创建模型
    if model_type == 'lite':
        model = FLARELite(in_channels=3, base_ch=base_ch)
    else:
        model = FLARE(in_channels=3, base_ch=base_ch)

    # 加载权重
    if model_path and os.path.exists(model_path):
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        st.sidebar.success(f"✓ 模型已加载: {os.path.basename(model_path)} (base_ch={base_ch})")
    else:
        st.sidebar.warning("⚠ 使用随机初始化权重！")

    # 确保模型在正确设备上
    model = model.to(device)
    st.sidebar.success(f"使用设备: {device.upper()}")

    engine = PyTorchEngine(model, img_size=DEFAULT_IMG_SIZE, device=device)
    anti_glare = AntiGlarePipeline()
    localizer = SubpixelLocalizer(
        min_area=100,
        compute_gripper=compute_gripper,
        gripper_width_px=gripper_width,
    )

    return engine, anti_glare, localizer


def process_image(
    engine,
    anti_glare,
    localizer,
    image_bgr: np.ndarray,
    apply_hdr: bool = True,
) -> dict:
    """处理单张图像。"""
    t_start = time.perf_counter()
    timing = {}

    # HDR 融合
    t0 = time.perf_counter()
    if apply_hdr:
        image_hdr = anti_glare.process_single(image_bgr)
    else:
        image_hdr = image_bgr.copy()
    timing['hdr_ms'] = (time.perf_counter() - t0) * 1000

    # 高光检测
    t0 = time.perf_counter()
    glare_mask_raw = detect_highlight_mask(image_hdr)
    timing['glare_ms'] = (time.perf_counter() - t0) * 1000

    # 推理
    t0 = time.perf_counter()
    seg_mask, edge_mask = engine.predict(image_hdr)
    timing['infer_ms'] = (time.perf_counter() - t0) * 1000

    # 定位
    t0 = time.perf_counter()
    gray = cv2.cvtColor(image_hdr, cv2.COLOR_BGR2GRAY)
    detections = localizer.localize(
        seg_mask, edge_mask, intensity_image=gray, glare_mask=glare_mask_raw
    )
    timing['localize_ms'] = (time.perf_counter() - t0) * 1000

    # 坐标变换
    for det in detections:
        det['position_robot_mm'] = (
            det['centroid_px'][0] * 2 - 512,
            det['centroid_px'][1] * 2 - 512,
            800.0,
        )

    timing['total_ms'] = (time.perf_counter() - t_start) * 1000

    return {
        'original': image_bgr,
        'processed': image_hdr,
        'seg_mask': seg_mask,
        'edge_mask': edge_mask,
        'glare_mask': glare_mask_raw,
        'detections': detections,
        'timing': timing,
    }


# ============================================================
# 3. 可视化工具
# ============================================================

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """绘制检测框和标签，包含机械抓取位置。"""
    result = image.copy()
    h, w = image.shape[:2]

    # 先绘制抓取配置（底层）
    if any('gripper_config' in det for det in detections):
        result = draw_all_grasps(result, detections, show_quality=True)

    # 再绘制检测信息
    for i, det in enumerate(detections):
        cx, cy = int(det['centroid_px'][0]), int(det['centroid_px'][1])
        angle = det['orientation_deg']
        ftype = det['feature_type']

        color = {'blob': (0, 255, 0), 'line': (255, 100, 0),
                 'region': (0, 165, 255)}.get(ftype, (255, 255, 255))

        cv2.circle(result, (cx, cy), 8, color, -1)
        cv2.circle(result, (cx, cy), 8, (255, 255, 255), 2)

        length = 40
        angle_rad = np.radians(angle)
        ex = int(cx + length * np.cos(angle_rad))
        ey = int(cy + length * np.sin(angle_rad))
        cv2.arrowedLine(result, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

        pos = det.get('position_robot_mm')
        if pos:
            label = f"[{i}] {ftype} ({pos[0]:.0f},{pos[1]:.0f})"
        else:
            label = f"[{i}] {ftype}"
        cv2.putText(result, label, (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result


def create_comparison_view(
    original: np.ndarray,
    processed: np.ndarray,
    seg_mask: np.ndarray,
    edge_mask: np.ndarray,
    result: np.ndarray,
) -> np.ndarray:
    """创建对比视图。"""
    h, w = original.shape[:2]

    # 缩放到相同尺寸
    seg_color = cv2.cvtColor(cv2.resize(seg_mask, (w, h)), cv2.COLOR_GRAY2BGR)
    edge_color = cv2.cvtColor(cv2.resize(edge_mask, (w, h)), cv2.COLOR_GRAY2BGR)

    # 上排：原图 vs 处理后
    top = np.hstack([original, processed])
    # 下排：分割 vs 检测结果
    bottom = np.hstack([seg_color, result])

    grid = np.vstack([top, bottom])
    return grid


def image_to_bytes(img: np.ndarray) -> bytes:
    """图像转 bytes 用于 Streamlit 显示。"""
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


# ============================================================
# 4. Streamlit UI
# ============================================================

def main():
    st.title("🔬 FLARE 视觉分拣系统")
    st.markdown("**Fast Light Anti-Reflective Engine** — 船舶高反光工件精准分拣 | YOLOv10-style Architecture")

    # ===== 侧边栏 =====
    with st.sidebar:
        st.header("⚙️ 配置")

        st.markdown("**FLARE** — Fast Light Anti-Reflective Engine")
        st.markdown("```\n参数量: 8.0M (standard) / 1.2M (lite)\n延迟:   ~10ms GPU / ~40ms CPU\nIoU:    0.41+ (合成数据)\n```")

        st.divider()
        model_type = st.selectbox(
            "模型",
            ["standard", "lite"],
            index=0,
            help="standard: 8.0M参数; lite: 1.2M参数"
        )

        base_ch = st.slider("基础通道数", 32, 128, 32, step=8,
                            help="越大精度越高，越慢（通常与训练时一致）")

        # 自动检测可用模型
        import glob
        project_root = Path(__file__).parent
        checkpoint_dirs = [
            project_root / "checkpoints",
            project_root / "checkpoints_new",
        ]
        available_models = {}
        for ckpt_dir in checkpoint_dirs:
            if ckpt_dir.exists():
                for pth_file in sorted(ckpt_dir.glob("*.pth")):
                    available_models[str(pth_file)] = pth_file.name

        model_options = ["（不使用模型-随机权重）"] + list(available_models.keys())
        default_idx = 1
        for i, opt in enumerate(model_options):
            if "best" in opt:
                default_idx = i
                break

        selected_model = st.selectbox(
            "模型文件",
            model_options,
            index=default_idx,
            help="选择训练好的模型权重"
        )
        model_path = None if selected_model.startswith("（不") else selected_model

        if model_path:
            st.success(f"已选择: {available_models.get(model_path, Path(model_path).name)}")
        else:
            st.warning("使用随机权重，识别效果会很差！")

        st.divider()
        st.header("🤖 抓取配置")
        compute_gripper = st.checkbox("计算机械抓取点", value=True,
                                       help="为每个检测目标计算最优抓取位置")
        gripper_width = st.slider("爪宽度 (px)", 20, 80, 40, step=5,
                                   help="机械爪张开宽度")

        apply_hdr = st.checkbox("启用 HDR 反光抑制", value=True)
        show_comparison = st.checkbox("显示对比视图", value=True)

        st.divider()

        # 性能指标
        st.header("📊 性能指标")
        fps_placeholder = st.empty()
        latency_placeholder = st.empty()

        st.divider()

        # 上传
        st.header("📤 上传")
        uploaded_file = st.file_uploader(
            "选择图像",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="支持单张图像或同时上传多张"
        )

        uploaded_video = st.file_uploader(
            "或选择视频",
            type=['mp4', 'avi', 'mov'],
            help="视频将逐帧处理"
        )

    # ===== 主界面 =====
    if uploaded_file is not None:
        # 读取图像
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("无法读取图像")
            return

        # 加载管线
        with st.spinner("加载模型..."):
            engine, anti_glare, localizer = load_pipeline(
                model_path, model_type, base_ch,
                compute_gripper, gripper_width,
            )

        # 处理
        with st.spinner("推理中..."):
            result = process_image(engine, anti_glare, localizer, image, apply_hdr)

        # 性能更新
        fps = 1000.0 / result['timing']['total_ms']
        fps_placeholder.metric("FPS", f"{fps:.1f}")
        latency_placeholder.metric("延迟", f"{result['timing']['total_ms']:.0f} ms")

        # 显示
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("原图 + 高光检测")
            glare_vis = image.copy()
            glare_vis[result['glare_mask'] > 0] = [0, 0, 220]
            st.image(cv2.cvtColor(glare_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("检测结果")
            result_vis = draw_detections(result['processed'], result['detections'])
            st.image(cv2.cvtColor(result_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

        # 对比视图
        if show_comparison:
            st.subheader("处理流程对比")
            grid = create_comparison_view(
                image,
                result['processed'],
                result['seg_mask'],
                result['edge_mask'],
                result_vis,
            )
            st.image(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), use_container_width=True)

        # 检测详情
        st.subheader("📋 检测报告")
        if result['detections']:
            # 统计
            n_total = len(result['detections'])
            n_blob = sum(1 for d in result['detections'] if d['feature_type'] == 'blob')
            n_line = sum(1 for d in result['detections'] if d['feature_type'] == 'line')
            n_region = sum(1 for d in result['detections'] if d['feature_type'] == 'region')

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总数", n_total)
            m2.metric("Blob", n_blob)
            m3.metric("Line", n_line)
            m4.metric("Region", n_region)

            # 详细表格
            st.markdown("#### 目标列表")
            data = []
            for i, det in enumerate(result['detections']):
                pos = det['position_robot_mm']
                gc = det.get('gripper_config', {})
                data.append({
                    "ID": i,
                    "类型": det['feature_type'],
                    "质心 X": f"{det['centroid_px'][0]:.1f}",
                    "质心 Y": f"{det['centroid_px'][1]:.1f}",
                    "方向°": f"{det['orientation_deg']:.1f}",
                    "面积 px²": f"{det['area_px']:.0f}",
                    "抓取质量": f"{gc.get('grip_quality', 0):.2f}" if gc else "N/A",
                    "爪宽 px": f"{gc.get('gripper_width_px', 0):.0f}" if gc else "N/A",
                    "接近角°": f"{gc.get('approach_angle_deg', 0):.1f}" if gc else "N/A",
                    "坐标 X": f"{pos[0]:.0f}",
                    "坐标 Y": f"{pos[1]:.0f}",
                    "坐标 Z": f"{pos[2]:.0f}",
                })
            st.dataframe(data, use_container_width=True)

            # 抓取详情
            gripper_dets = [d for d in result['detections'] if 'gripper_config' in d]
            if gripper_dets:
                with st.expander("🤖 机械抓取配置详情"):
                    st.markdown("#### 抓取接触点")
                    gripper_data = []
                    for i, det in enumerate(gripper_dets):
                        gc = det['gripper_config']
                        gripper_data.append({
                            "ID": i,
                            "左爪 X": f"{gc['left_contact_px'][0]:.1f}",
                            "左爪 Y": f"{gc['left_contact_px'][1]:.1f}",
                            "右爪 X": f"{gc['right_contact_px'][0]:.1f}",
                            "右爪 Y": f"{gc['right_contact_px'][1]:.1f}",
                            "抓取中心 X": f"{gc['center_px'][0]:.1f}",
                            "抓取中心 Y": f"{gc['center_px'][1]:.1f}",
                            "质量": f"{gc['grip_quality']:.3f}",
                        })
                    st.dataframe(gripper_data, use_container_width=True)

                    # 可视化说明
                    st.markdown("""
                    **图例说明**:
                    - 🔴 红色圆点 = 左爪接触点
                    - 🟢 绿色圆点 = 右爪接触点
                    - 🔵 青色圆点 = 抓取中心
                    - 🟡 黄色线 = 抓取宽度
                    - 🟣 紫色箭头 = 接近方向
                    """)

            # 下载按钮
            if st.button("📥 导出 CSV"):
                import pandas as pd
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "下载 CSV",
                    csv,
                    "detections.csv",
                    "text/csv",
                )
        else:
            st.info("未检测到目标")

        # 处理时间分解
        with st.expander("⏱️ 处理时间分解"):
            timing = result['timing']
            st.markdown(f"""
            | 阶段 | 耗时 |
            |------|------|
            | HDR融合 | {timing['hdr_ms']:.1f} ms |
            | 高光检测 | {timing['glare_ms']:.1f} ms |
            | 模型推理 | {timing['infer_ms']:.1f} ms |
            | 像素级定位 | {timing['localize_ms']:.1f} ms |
            | **总计** | **{timing['total_ms']:.1f} ms** |
            """)

    elif uploaded_video is not None:
        st.info("视频处理功能开发中...")
        st.warning("请使用图像上传进行演示")

    else:
        # 欢迎页
        st.markdown("""
        ### 👋 欢迎使用 FLARE 演示系统

        **上传图像** 开始检测，或在侧边栏配置参数。

        #### 功能特性
        - 🔦 **HDR 反光抑制** — 多曝光融合，高光区域修复
        - 🎯 **像素级边缘定位** — 灰度加权矩 + PCA 主轴方向估计
        - 🤖 **FLARE 深度学习** — WaveletScattering + FourierConv + CBAM 注意力
        - 📊 **实时性能监控** — FPS / 延迟仪表盘

        #### 支持的场景
        - 🚢 船舶大型钢板高反光工件
        - 🏭 汽车门板钢、铝合金
        - 🌉 桥梁、高铁等大型金属结构
        """)

        # 示例图像
        st.divider()
        st.subheader("🖼️ 快速演示")

        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            st.info("上传图像以开始演示")
        with demo_col2:
            st.info("配置参数后上传图像")


if __name__ == "__main__":
    main()
