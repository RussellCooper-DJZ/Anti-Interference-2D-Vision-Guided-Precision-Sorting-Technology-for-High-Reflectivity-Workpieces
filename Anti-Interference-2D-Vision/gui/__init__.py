"""
gui — AGEANet 原生桌面 GUI 包

基于 PyQt6 的跨平台桌面应用程序，提供：
- 图像拖拽/选择
- 模型加载与切换
- 参数实时调整
- 分割/边缘结果可视化
- 结果导出（PNG/JPG/CSV）

依赖：
    pip install -r requirements-gui.txt

启动：
    python gui/main_window.py
"""

__all__ = ["AGEANetGUI", "main"]
