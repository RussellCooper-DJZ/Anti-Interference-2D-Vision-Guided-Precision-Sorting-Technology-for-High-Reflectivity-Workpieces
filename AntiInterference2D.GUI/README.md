# AntiInterference2D.GUI

## AGEANet 高反光工件视觉检测系统 — C# WPF 上位机

### 技术栈

- **UI 框架**：WPF (.NET 8)
- **架构模式**：MVVM (CommunityToolkit.Mvvm)
- **通信方式**：HTTP REST API → Python FastAPI 后端
- **图像处理**：WPF Imaging (WriteableBitmap / BitmapImage)

### 项目结构

```
AntiInterference2D.GUI/
├── Assets/
│   └── Styles.xaml              # 全局样式资源
├── Converters/
│   ├── BoolToColorConverter.cs  # 布尔转颜色（连接状态指示）
│   ├── InverseBooleanConverter.cs
│   └── NullToVisibilityConverter.cs
├── Models/
│   └── SystemModels.cs          # 数据模型（推理结果、配置等）
├── Services/
│   ├── ApiClient.cs             # FastAPI HTTP 客户端
│   ├── ImageProcessingService.cs # 图像编解码辅助
│   └── RobotCommunicationService.cs # ABB TCP 通信
├── ViewModels/
│   ├── MainWindowViewModel.cs   # 主窗口 VM（导航管理）
│   ├── DashboardViewModel.cs
│   ├── ImageInspectionViewModel.cs
│   ├── BatchProcessingViewModel.cs
│   ├── ParameterConfigViewModel.cs
│   ├── ModelManagementViewModel.cs
│   └── RobotControlViewModel.cs
├── Views/
│   ├── MainWindow.xaml          # 主窗口（左侧导航 + 内容区）
│   ├── DashboardView.xaml
│   ├── ImageInspectionView.xaml # 四宫格图像检测
│   ├── BatchProcessingView.xaml
│   ├── ParameterConfigView.xaml
│   ├── ModelManagementView.xaml
│   └── RobotControlView.xaml
├── App.xaml / App.xaml.cs       # 应用入口 + DI 容器
└── AntiInterference2D.GUI.csproj
```

### 功能模块

| 模块 | 功能 |
|------|------|
| 仪表盘 | 系统状态总览（模型/设备/显存） |
| 图像检测 | 单张图像上传 → HDR/高光修复 → 推理 → 四宫格可视化 |
| 批量处理 | 文件夹批量推理 → 结果列表 → CSV 导出 |
| 参数配置 | 阈值滑块实时调参、设备/后端切换 |
| 模型管理 | 扫描 checkpoints → 加载/切换模型 |
| 机器人控制 | ABB RobotStudio / EGM 连接、目标位姿发送 |

### 快速开始

#### 前提条件

1. **Python 后端已启动**（见 `../Anti-Interference-2D-Vision/start_api.bat`）
2. **.NET 8 SDK** 已安装

#### 编译运行

```bash
cd AntiInterference2D.GUI
dotnet restore
dotnet build
dotnet run
```

或在 Visual Studio 中打开 `.csproj` 文件，按 F5 运行。

### 后端 API 端点

上位机通过以下端点与 Python 后端通信：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/status` | GET | 系统状态 |
| `/api/v1/models` | GET | 列出模型 |
| `/api/v1/models/switch` | POST | 切换模型 |
| `/api/v1/config` | GET/POST | 配置管理 |
| `/api/v1/infer/image` | POST | 单张图像推理 |
| `/api/v1/infer/batch` | POST | 批量推理 |
| `/api/v1/infer/base64` | POST | Base64 图像推理 |
