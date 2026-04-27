#!/usr/bin/env python3
"""
gui/main_window.py — AGEANet 原生桌面 GUI

基于 PyQt6，功能：
  - 图像拖拽上传 / 文件选择
  - FLARE / FLARELite 模型推理
  - HDR 预处理开关
  - 分割/边缘/高光掩膜可视化
  - 结果导出

启动：
    python gui/main_window.py [--model_path checkpoints/best.pth]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QComboBox, QFileDialog, QMessageBox,
        QProgressBar, QGroupBox, QSplitter, QTextEdit, QCheckBox, QSpinBox,
        QDoubleSpinBox, QStatusBar,
    )
except ImportError as e:
    print("[错误] 缺少 PyQt6，请先安装 GUI 依赖:")
    print("  pip install -r requirements-gui.txt")
    raise SystemExit(1)

import cv2
import torch

from vision.feature_extraction import FLARE, FLARELite, predict
from vision.hdr_processing import AntiGlarePipeline, detect_highlight_mask
from vision.localization_and_calibration import SubpixelLocalizer


# ============================================================
# 推理工作线程（避免阻塞 UI）
# ============================================================

class InferenceWorker(QThread):
    finished = pyqtSignal(dict)   # 推理结果
    error = pyqtSignal(str)       # 错误信息
    progress = pyqtSignal(int)    # 进度 0-100

    def __init__(self, image: np.ndarray, model, use_hdr: bool = True,
                 device: str = "cpu"):
        super().__init__()
        self.image = image
        self.model = model
        self.use_hdr = use_hdr
        self.device = device

    def run(self):
        try:
            self.progress.emit(10)

            # HDR 预处理
            if self.use_hdr:
                pipeline = AntiGlarePipeline()
                processed = pipeline.process(self.image)
            else:
                processed = self.image.copy()

            self.progress.emit(40)

            # 推理
            result = predict(self.model, processed, device=self.device)
            seg_mask = result.get("seg")
            edge_mask = result.get("edge")

            self.progress.emit(70)

            # 高光检测
            glare_mask = detect_highlight_mask(processed)

            self.progress.emit(90)

            # 组装结果
            output = {
                "original": self.image,
                "processed": processed,
                "seg_mask": seg_mask,
                "edge_mask": edge_mask,
                "glare_mask": glare_mask,
            }

            self.progress.emit(100)
            self.finished.emit(output)

        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# 图像显示标签（支持拖拽）
# ============================================================

class ImageLabel(QLabel):
    imageDropped = pyqtSignal(str)

    def __init__(self, title: str = "图像"):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(f"[{title}]\n拖拽图片到这里\n或点击右侧按钮选择")
        self.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #aaaaaa;
                border: 2px dashed #555555;
                border-radius: 8px;
                font-size: 14px;
            }
        """)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 400)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.imageDropped.emit(path)

    def set_image(self, img: np.ndarray):
        """设置并显示 OpenCV 图像 (H, W, C) uint8"""
        if img is None:
            return
        if len(img.shape) == 2:
            # 灰度
            h, w = img.shape
            bytes_per_line = w
            q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        # 缩放以适应标签大小，保持比例
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #444444;
                border-radius: 8px;
            }
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 如果当前有 pixmap，重新缩放
        if self.pixmap():
            scaled = self.pixmap().scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


# ============================================================
# 主窗口
# ============================================================

class AGEANetGUI(QMainWindow):
    def __init__(self, model_path: str = None, model_type: str = "standard"):
        super().__init__()
        self.setWindowTitle("AGEANet — 抗干扰 2D 视觉分拣系统")
        self.setMinimumSize(1400, 900)

        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.current_image: np.ndarray = None
        self.current_result: dict = None
        self.worker: InferenceWorker = None

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 左侧：控制面板
        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel, 0)

        # 右侧：图像显示区域
        display_area = self._build_display_area()
        main_layout.addWidget(display_area, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 请加载图片或拖拽到窗口")

        # 样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: #eeeeee;
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00d4aa;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5c5f;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                background: #00d4aa;
                border-radius: 8px;
                margin: -5px 0;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00d4aa;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #252525;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #cccccc;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 1px solid #555555;
            }
            QCheckBox::indicator:checked {
                background-color: #00d4aa;
                border: 1px solid #00d4aa;
            }
        """)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # 标题
        title = QLabel("🧠 AGEANet 控制面板")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)

        # === 模型设置 ===
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()

        # 模型类型
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("模型类型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["FLARE (标准)", "FLARELite (轻量)"])
        self.model_combo.currentIndexChanged.connect(self._on_model_type_changed)
        type_row.addWidget(self.model_combo)
        model_layout.addLayout(type_row)

        # 模型路径
        path_row = QHBoxLayout()
        self.model_path_label = QLabel(self.model_path or "未加载")
        self.model_path_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.model_path_label.setWordWrap(True)
        path_row.addWidget(QLabel("权重:"))
        path_row.addWidget(self.model_path_label, 1)
        model_layout.addLayout(path_row)

        btn_load_model = QPushButton("加载模型权重...")
        btn_load_model.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_load_model)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # === 预处理设置 ===
        pre_group = QGroupBox("预处理")
        pre_layout = QVBoxLayout()

        self.hdr_checkbox = QCheckBox("启用 HDR 反光抑制")
        self.hdr_checkbox.setChecked(True)
        pre_layout.addWidget(self.hdr_checkbox)

        self.highlight_checkbox = QCheckBox("显示高光掩膜")
        self.highlight_checkbox.setChecked(False)
        pre_layout.addWidget(self.highlight_checkbox)

        pre_group.setLayout(pre_layout)
        layout.addWidget(pre_group)

        # === 推理设置 ===
        infer_group = QGroupBox("推理参数")
        infer_layout = QVBoxLayout()

        # 边缘阈值
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("边缘阈值:"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 0.99)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setValue(0.40)
        thresh_row.addWidget(self.thresh_spin)
        infer_layout.addLayout(thresh_row)

        # 设备选择
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("运行设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (GPU)"])
        if not torch.cuda.is_available():
            self.device_combo.setEnabled(False)
            self.device_combo.setCurrentText("CPU")
        device_row.addWidget(self.device_combo)
        infer_layout.addLayout(device_row)

        infer_group.setLayout(infer_layout)
        layout.addWidget(infer_group)

        # === 操作按钮 ===
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout()

        btn_select = QPushButton("📂 选择图片")
        btn_select.clicked.connect(self._browse_image)
        action_layout.addWidget(btn_select)

        self.btn_run = QPushButton("▶️ 开始推理")
        self.btn_run.clicked.connect(self._run_inference)
        self.btn_run.setEnabled(False)
        action_layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        action_layout.addWidget(self.progress)

        btn_export = QPushButton("💾 导出结果")
        btn_export.clicked.connect(self._export_result)
        action_layout.addWidget(btn_export)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # === 日志 ===
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(200)
        self.log_edit.setPlaceholderText("日志输出...")
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, 1)

        # 底部信息
        info = QLabel("AGEANet v0.5.0 | Apache-2.0")
        info.setStyleSheet("color: #666666; font-size: 11px; margin-top: 8px;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        return panel

    def _build_display_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setSpacing(8)

        # 顶部标签栏
        tabs = QHBoxLayout()
        self.tab_original = QPushButton("原始图像")
        self.tab_processed = QPushButton("HDR 处理")
        self.tab_seg = QPushButton("分割结果")
        self.tab_edge = QPushButton("边缘检测")
        for btn in [self.tab_original, self.tab_processed, self.tab_seg, self.tab_edge]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2b2b2b;
                    color: #aaaaaa;
                    border: 1px solid #444444;
                    border-radius: 4px 4px 0 0;
                    padding: 6px 16px;
                }
                QPushButton:checked {
                    background-color: #0d7377;
                    color: white;
                    border-bottom: none;
                }
            """)
            btn.clicked.connect(self._on_tab_changed)
            tabs.addWidget(btn)
        tabs.addStretch(1)
        layout.addLayout(tabs)

        self.tab_buttons = [self.tab_original, self.tab_processed, self.tab_seg, self.tab_edge]
        self.tab_original.setChecked(True)

        # 图像显示
        self.image_label = ImageLabel("等待图片")
        layout.addWidget(self.image_label, 1)

        # 信息显示
        self.info_label = QLabel("分辨率: - | 格式: -")
        self.info_label.setStyleSheet("color: #888888; font-size: 12px; padding: 4px;")
        layout.addWidget(self.info_label)

        return area

    def _log(self, msg: str):
        self.log_edit.append(f"[{QApplication.instance().applicationName()}] {msg}")

    def _on_model_type_changed(self, index: int):
        self.model_type = "standard" if index == 0 else "lite"
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            device = "cuda" if torch.cuda.is_available() and self.device_combo.currentText().startswith("CUDA") else "cpu"

            if self.model_type == "lite":
                self.model = FLARELite(in_channels=3, base_ch=32)
            else:
                self.model = FLARE(in_channels=3, base_ch=64)

            self.model.to(device)
            self.model.eval()

            if self.model_path and os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=device)
                self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
                self._log(f"模型加载成功: {self.model_path}")
            else:
                self._log(f"使用随机初始化模型 (未找到权重: {self.model_path})")

            self.status_bar.showMessage(f"模型就绪 | 设备: {device.upper()} | 类型: {self.model_type}")

        except Exception as e:
            self._log(f"模型加载失败: {e}")
            self.status_bar.showMessage(f"模型加载失败: {e}")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", str(PROJECT_ROOT / "checkpoints"),
            "PyTorch 权重 (*.pth *.pt);;所有文件 (*.*)"
        )
        if path:
            self.model_path = path
            self.model_path_label.setText(os.path.basename(path))
            self._load_model()

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*.*)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载图片: {path}")
            return

        self.current_image = img
        self.current_result = None
        self.image_label.set_image(img)

        h, w = img.shape[:2]
        self.info_label.setText(f"分辨率: {w}×{h} | 格式: {Path(path).suffix.upper()}")
        self.status_bar.showMessage(f"图片已加载: {os.path.basename(path)} ({w}×{h})")
        self.btn_run.setEnabled(True)
        self._log(f"加载图片: {path}")

    def _run_inference(self):
        if self.current_image is None or self.model is None:
            return

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.status_bar.showMessage("推理中...")
        self._log("开始推理...")

        device = "cuda" if self.device_combo.currentText().startswith("CUDA") else "cpu"

        self.worker = InferenceWorker(
            self.current_image, self.model,
            use_hdr=self.hdr_checkbox.isChecked(),
            device=device,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_inference_finished)
        self.worker.error.connect(self._on_inference_error)
        self.worker.start()

    def _on_inference_finished(self, result: dict):
        self.current_result = result
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        self.status_bar.showMessage("推理完成")
        self._log("推理完成")

        # 默认显示分割结果
        self.tab_seg.setChecked(True)
        self._on_tab_changed()

    def _on_inference_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress.setValue(0)
        QMessageBox.critical(self, "推理错误", msg)
        self._log(f"推理错误: {msg}")
        self.status_bar.showMessage(f"推理失败: {msg}")

    def _on_tab_changed(self):
        sender = self.sender()
        if sender:
            for btn in self.tab_buttons:
                if btn != sender:
                    btn.setChecked(False)
            sender.setChecked(True)

        if self.current_result is None:
            return

        # 根据当前选中的标签显示对应图像
        if self.tab_original.isChecked():
            self.image_label.set_image(self.current_result["original"])
        elif self.tab_processed.isChecked():
            self.image_label.set_image(self.current_result["processed"])
        elif self.tab_seg.isChecked():
            seg = self.current_result["seg_mask"]
            if seg is not None:
                seg_vis = (seg * 255).astype(np.uint8)
                # 伪彩色
                seg_color = cv2.applyColorMap(seg_vis, cv2.COLORMAP_JET)
                self.image_label.set_image(seg_color)
        elif self.tab_edge.isChecked():
            edge = self.current_result["edge_mask"]
            if edge is not None:
                edge_vis = (edge * 255).astype(np.uint8)
                self.image_label.set_image(edge_vis)

    def _export_result(self):
        if self.current_result is None:
            QMessageBox.information(self, "提示", "请先运行推理")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not folder:
            return

        base = Path(folder)
        try:
            cv2.imwrite(str(base / "original.png"), self.current_result["original"])
            cv2.imwrite(str(base / "processed.png"), self.current_result["processed"])

            seg = self.current_result["seg_mask"]
            if seg is not None:
                seg_vis = (seg * 255).astype(np.uint8)
                cv2.imwrite(str(base / "segmentation.png"), seg_vis)

            edge = self.current_result["edge_mask"]
            if edge is not None:
                edge_vis = (edge * 255).astype(np.uint8)
                cv2.imwrite(str(base / "edge.png"), edge_vis)

            glare = self.current_result["glare_mask"]
            if glare is not None:
                cv2.imwrite(str(base / "glare_mask.png"), glare)

            self._log(f"结果已导出到: {folder}")
            self.status_bar.showMessage(f"导出完成: {folder}")
            QMessageBox.information(self, "导出成功", f"结果已保存到:\n{folder}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AGEANet 原生桌面 GUI")
    parser.add_argument("--model_path", default=None, help="模型权重路径")
    parser.add_argument("--model_type", default="standard", choices=["standard", "lite"], help="模型类型")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("AGEANet")
    app.setApplicationVersion("0.5.0")

    window = AGEANetGUI(model_path=args.model_path, model_type=args.model_type)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
