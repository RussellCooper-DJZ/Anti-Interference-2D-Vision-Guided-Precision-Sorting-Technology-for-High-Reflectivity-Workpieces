"""
api/server.py — FastAPI 后端服务
为 C# WPF 上位机提供 RESTful API 接口

端点一览：
  POST /api/v1/infer/image      单张图像推理
  POST /api/v1/infer/batch      批量图像推理
  GET  /api/v1/models           列出可用模型
  POST /api/v1/models/switch    切换当前模型
  GET  /api/v1/config           获取推理配置
  POST /api/v1/config           更新推理配置
  GET  /api/v1/health           健康检查
  GET  /api/v1/status           系统状态（GPU/内存/模型信息）

启动方式::
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# FastAPI
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 项目内部导入（假设从 api/ 启动时，.. 指向项目根）
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.feature_extraction import FLARE, FLARELite, get_model_info
from vision.hdr_processing import AntiGlarePipeline, detect_highlight_mask, repair_highlight_regions
from vision.localization_and_calibration import SubpixelLocalizer, CoordinateTransformer
from vision.inference_engine import create_engine, BaseInferenceEngine

# ------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api.server")

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI(
    title="AGEANet API Server",
    description="高反光工件视觉检测后端服务（供 C# WPF 上位机调用）",
    version="1.0.0",
)

# CORS：允许本地 WPF 应用访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为 localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 全局状态
# ------------------------------------------------------------------

class ServerState:
    """服务器运行时状态（单例）"""

    def __init__(self):
        self.engine: Optional[BaseInferenceEngine] = None
        self.current_model_path: Optional[str] = None
        self.current_model_type: str = "FLARE"  # FLARE / FLARELite / ONNX / TensorRT
        self.config = InferenceConfig()
        self.localizer = SubpixelLocalizer()
        self.coordinate_transformer: Optional[CoordinateTransformer] = None
        self._lock = False  # 简单推理锁，防止并发模型切换导致崩溃

    def is_ready(self) -> bool:
        return self.engine is not None


# ------------------------------------------------------------------
# Pydantic 模型
# ------------------------------------------------------------------

class InferenceConfig(BaseModel):
    """推理配置参数"""

    img_size: int = Field(512, ge=128, le=2048, description="输入图像尺寸")
    use_hdr: bool = Field(True, description="是否启用 HDR 预处理")
    use_highlight_repair: bool = Field(True, description="是否修复高光区域")
    seg_threshold: float = Field(0.5, ge=0.0, le=1.0, description="分割阈值")
    edge_threshold: float = Field(0.3, ge=0.0, le=1.0, description="边缘阈值")
    device: str = Field("auto", description="推理设备：auto/cpu/cuda")
    backend: str = Field("pytorch", description="推理后端：pytorch/onnx/tensorrt")
    return_visualization: bool = Field(True, description="是否返回可视化结果图")
    return_coordinates: bool = Field(True, description="是否返回定位坐标")


class ModelInfo(BaseModel):
    name: str
    path: str
    type: str
    model_arch: str
    size_mb: float
    loaded: bool


class InferResult(BaseModel):
    success: bool
    message: str
    latency_ms: float
    seg_mask_b64: Optional[str] = None
    edge_mask_b64: Optional[str] = None
    highlight_mask_b64: Optional[str] = None
    vis_image_b64: Optional[str] = None
    coordinates: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, float]] = None


class SystemStatus(BaseModel):
    ready: bool
    current_model: Optional[str]
    current_model_type: str
    device: str
    cuda_available: bool
    cuda_device_name: Optional[str]
    memory_used_mb: float
    memory_total_mb: float


# Global state singleton (must be defined after InferenceConfig)
state = ServerState()


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """OpenCV BGR → PIL RGB"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _encode_image_b64(cv_img: np.ndarray, fmt: str = ".png") -> str:
    """OpenCV 图像编码为 Base64 字符串"""
    success, buf = cv2.imencode(fmt, cv_img)
    if not success:
        raise ValueError("图像编码失败")
    return base64.b64encode(buf).decode("utf-8")


def _decode_image_b64(b64_str: str) -> np.ndarray:
    """Base64 字符串解码为 OpenCV BGR 图像"""
    buf = base64.b64decode(b64_str)
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图像解码失败")
    return img


def _create_visualization(
    original: np.ndarray,
    seg_mask: np.ndarray,
    edge_mask: np.ndarray,
    highlight_mask: Optional[np.ndarray] = None,
    coordinates: Optional[List[Dict]] = None,
) -> np.ndarray:
    """创建四宫格可视化图像"""
    h, w = original.shape[:2]
    grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # 左上：原图 + 高光标注
    top_left = original.copy()
    if highlight_mask is not None:
        red_overlay = np.zeros_like(top_left)
        red_overlay[highlight_mask > 0] = (0, 0, 255)
        top_left = cv2.addWeighted(top_left, 0.7, red_overlay, 0.3, 0)
    grid[0:h, 0:w] = top_left

    # 右上：分割结果（绿色半透明叠加）
    top_right = original.copy()
    seg_overlay = np.zeros_like(top_right)
    seg_overlay[seg_mask > 0] = (0, 255, 0)
    grid[0:h, w : 2 * w] = cv2.addWeighted(top_right, 0.6, seg_overlay, 0.4, 0)

    # 左下：边缘检测（青色叠加）
    bottom_left = original.copy()
    edge_overlay = np.zeros_like(bottom_left)
    edge_overlay[edge_mask > 0] = (255, 255, 0)
    grid[h : 2 * h, 0:w] = cv2.addWeighted(bottom_left, 0.6, edge_overlay, 0.4, 0)

    # 右下：定位结果
    bottom_right = original.copy()
    if coordinates:
        for coord in coordinates:
            cx, cy = int(coord.get("cx", 0)), int(coord.get("cy", 0))
            if cx == 0 and cy == 0:
                continue
            cv2.circle(bottom_right, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                bottom_right,
                f"({coord.get('x_mm', 0):.1f},{coord.get('y_mm', 0):.1f})",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
    grid[h : 2 * h, w : 2 * w] = bottom_right

    return grid


def _discover_models(checkpoint_dir: str = "./checkpoints") -> List[ModelInfo]:
    """扫描 checkpoints 目录，发现可用模型"""
    models: List[ModelInfo] = []
    cp = Path(checkpoint_dir)
    if not cp.exists():
        return models

    for p in cp.rglob("*"):
        if p.suffix in (".pth", ".pt", ".onnx", ".engine"):
            size_mb = p.stat().st_size / (1024 * 1024)
            model_type = (
                "ONNX" if p.suffix == ".onnx" else "TensorRT" if p.suffix == ".engine" else "PyTorch"
            )
            # 根据文件名推断模型架构
            stem_lower = p.stem.lower()
            if "lite" in stem_lower or "small" in stem_lower or "mobile" in stem_lower:
                model_arch = "FLARELite"
            else:
                model_arch = "FLARE"
            loaded = state.current_model_path == str(p)
            models.append(
                ModelInfo(
                    name=p.stem,
                    path=str(p),
                    type=model_type,
                    model_arch=model_arch,
                    size_mb=round(size_mb, 2),
                    loaded=loaded,
                )
            )
    return models


# ------------------------------------------------------------------
# 引擎初始化
# ------------------------------------------------------------------

def _load_model(model_path: str, model_type: str = "FLARE", config: InferenceConfig = None):
    """加载模型到推理引擎"""
    if config is None:
        config = state.config

    logger.info(f"Loading model: {model_path} (type={model_type}, backend={config.backend})")

    if config.backend in ("onnx", "tensorrt"):
        engine = create_engine(model_path, backend=config.backend)
    else:
        import torch

        if config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)
        if model_type == "FLARELite":
            model = FLARELite(in_channels=3, base_ch=32)
        else:
            model = FLARE(in_channels=3, base_ch=64)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        # 兼容旧版 checkpoint（键名可能因迭代版本不同而略有差异）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys (loaded with strict=False): {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys (ignored): {unexpected}")

        from vision.inference_engine import PyTorchEngine

        engine = PyTorchEngine(model, img_size=config.img_size, device=str(device))

    state.engine = engine
    state.current_model_path = model_path
    state.current_model_type = model_type
    logger.info("Model loaded successfully.")


# ------------------------------------------------------------------
# API 端点
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "AGEANet API Server", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "ready": state.is_ready()}


@app.get("/api/v1/status", response_model=SystemStatus)
def status():
    import torch

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    mem_used = mem_total = 0.0
    if cuda_available:
        mem_used = torch.cuda.memory_allocated() / 1024 / 1024
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    return SystemStatus(
        ready=state.is_ready(),
        current_model=state.current_model_path,
        current_model_type=state.current_model_type,
        device=state.config.device,
        cuda_available=cuda_available,
        cuda_device_name=device_name,
        memory_used_mb=round(mem_used, 1),
        memory_total_mb=round(mem_total, 1),
    )


@app.get("/api/v1/models", response_model=List[ModelInfo])
def list_models(checkpoint_dir: str = "./checkpoints"):
    return _discover_models(checkpoint_dir)


@app.post("/api/v1/models/switch")
def switch_model(model_path: str = Form(...), model_type: str = Form("FLARE"), model_arch: str = Form(None)):
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {model_path}")
    try:
        arch = model_arch if model_arch else model_type
        _load_model(model_path, arch)
        return {"success": True, "message": "模型切换成功", "model": model_path}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/config")
def get_config():
    return state.config.dict()


@app.post("/api/v1/config")
def update_config(cfg: InferenceConfig):
    state.config = cfg
    return {"success": True, "config": cfg.dict()}


@app.post("/api/v1/infer/image", response_model=InferResult)
def infer_image(
    file: UploadFile = File(...),
    use_hdr: bool = Form(True),
    use_highlight_repair: bool = Form(True),
    seg_threshold: float = Form(0.5),
    edge_threshold: float = Form(0.3),
    return_visualization: bool = Form(True),
    return_coordinates: bool = Form(True),
):
    """
    单张图像推理端点。
    上传图像文件，返回分割/边缘/高光/可视化/坐标结果。
    """
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="模型未加载，请先调用 /api/v1/models/switch")

    if state._lock:
        raise HTTPException(status_code=429, detail="服务器正忙，请稍后重试")

    t_start = time.perf_counter()
    state._lock = True
    try:
        # 1. 读取图像
        contents = file.file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="无法解析上传的图像")

        # 2. HDR / 高光预处理
        highlight_mask = None
        if use_hdr:
            pipeline = AntiGlarePipeline()
            img_bgr = pipeline.process_single(img_bgr)
        if use_highlight_repair:
            highlight_mask = detect_highlight_mask(img_bgr)
            if highlight_mask.sum() > 0:
                img_bgr = repair_highlight_regions(img_bgr, highlight_mask)

        # 3. 推理
        seg_mask, edge_mask = state.engine.predict(img_bgr)

        # 4. 后处理（阈值）
        seg_mask_bin = (seg_mask > (seg_threshold * 255)).astype(np.uint8) * 255
        edge_mask_bin = (edge_mask > (edge_threshold * 255)).astype(np.uint8) * 255

        # 5. 亚像素定位
        coordinates = None
        if return_coordinates:
            coords = state.localizer.localize(seg_mask_bin)
            coordinates = []
            for c in coords:
                centroid = c.get("centroid_px", (0, 0))
                cx = float(centroid[0]) if isinstance(centroid, (tuple, list)) else 0.0
                cy = float(centroid[1]) if isinstance(centroid, (tuple, list)) else 0.0
                coordinates.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "area": float(c.get("area_px", 0)),
                        "orientation": float(c.get("orientation_deg", 0)),
                        "x_mm": 0.0,
                        "y_mm": 0.0,
                        "z_mm": 0.0,
                    }
                )

        # 6. 可视化
        vis_b64 = None
        if return_visualization:
            vis_img = _create_visualization(
                img_bgr, seg_mask_bin, edge_mask_bin, highlight_mask, coordinates
            )
            vis_b64 = _encode_image_b64(vis_img)

        latency_ms = (time.perf_counter() - t_start) * 1000

        return InferResult(
            success=True,
            message="推理成功",
            latency_ms=round(latency_ms, 2),
            seg_mask_b64=_encode_image_b64(seg_mask_bin) if return_visualization else None,
            edge_mask_b64=_encode_image_b64(edge_mask_bin) if return_visualization else None,
            highlight_mask_b64=_encode_image_b64(highlight_mask) if (highlight_mask is not None and return_visualization) else None,
            vis_image_b64=vis_b64,
            coordinates=coordinates,
            metrics={
                "seg_ratio": float(seg_mask_bin.sum() / (seg_mask_bin.size * 255)),
                "edge_ratio": float(edge_mask_bin.sum() / (edge_mask_bin.size * 255)),
                "highlight_ratio": float(highlight_mask.sum() / highlight_mask.size) if highlight_mask is not None else 0.0,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        state._lock = False


@app.post("/api/v1/infer/batch")
def infer_batch(
    files: List[UploadFile] = File(...),
    use_hdr: bool = Form(True),
    seg_threshold: float = Form(0.5),
    edge_threshold: float = Form(0.3),
):
    """
    批量图像推理端点。
    返回每张图像的简要结果（不含大图 Base64，避免响应过大）。
    """
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="模型未加载")

    results = []
    for f in files:
        try:
            contents = f.file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                results.append({"filename": f.filename, "success": False, "error": "图像解码失败"})
                continue

            if use_hdr:
                pipeline = AntiGlarePipeline()
                img_bgr = pipeline.process_single(img_bgr)

            seg_mask, edge_mask = state.engine.predict(img_bgr)
            seg_mask_bin = (seg_mask > (seg_threshold * 255)).astype(np.uint8) * 255
            edge_mask_bin = (edge_mask > (edge_threshold * 255)).astype(np.uint8) * 255

            results.append(
                {
                    "filename": f.filename,
                    "success": True,
                    "seg_ratio": float(seg_mask_bin.sum() / (seg_mask_bin.size * 255)),
                    "edge_ratio": float(edge_mask_bin.sum() / (edge_mask_bin.size * 255)),
                }
            )
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})

    return {"success": True, "results": results}


@app.post("/api/v1/infer/base64")
def infer_base64(
    image_b64: str = Form(...),
    use_hdr: bool = Form(True),
    seg_threshold: float = Form(0.5),
    edge_threshold: float = Form(0.3),
    return_visualization: bool = Form(True),
):
    """
    Base64 图像推理端点（供 C# 客户端直接传输位图）。
    """
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        img_bgr = _decode_image_b64(image_b64)
        if use_hdr:
            pipeline = AntiGlarePipeline()
            img_bgr = pipeline.process_single(img_bgr)

        seg_mask, edge_mask = state.engine.predict(img_bgr)
        seg_mask_bin = (seg_mask > (seg_threshold * 255)).astype(np.uint8) * 255
        edge_mask_bin = (edge_mask > (edge_threshold * 255)).astype(np.uint8) * 255

        result = {
            "success": True,
            "seg_mask_b64": _encode_image_b64(seg_mask_bin) if return_visualization else None,
            "edge_mask_b64": _encode_image_b64(edge_mask_bin) if return_visualization else None,
        }

        if return_visualization:
            vis = _create_visualization(img_bgr, seg_mask_bin, edge_mask_bin)
            result["vis_image_b64"] = _encode_image_b64(vis)

        return result
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 启动时自动加载默认模型（如果存在）
# ------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    default_model = PROJECT_ROOT / "checkpoints" / "best.pth"
    if default_model.exists():
        try:
            _load_model(str(default_model), "FLARE")
        except Exception as e:
            logger.warning(f"自动加载默认模型失败: {e}")
    else:
        logger.info("默认模型不存在，等待手动加载。")


# ------------------------------------------------------------------
# 主入口（直接运行此文件时）
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
