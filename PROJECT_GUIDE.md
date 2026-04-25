# Anti-Interference-2D 项目目录指引

> 高反光工件抗干扰 2D 视觉引导精准分拣技术 —— 项目工作区总览

---

## 目录结构速查

```
Anti-Interference-2D/
├── Anti-Interference-2D-Vision/    ← 核心代码仓库（GitHub 已关联）
├── assets/                          ← 数据资产（测试图、视频）
├── results/                         ← 运行结果与产出
├── scripts/                         ← 根目录级运行脚本
├── competition/                     ← 比赛资料与评审文档
├── team/                            ← 团队提交资料
├── archive/                         ← 归档/历史版本
└── PROJECT_GUIDE.md                 ← 本文件
```

---

## 各目录说明

### 1. Anti-Interference-2D-Vision/（核心代码仓库）
**用途**：项目主代码仓库，已与 GitHub 远程仓库关联。

| 子目录 | 说明 |
|--------|------|
| `vision/` | 核心视觉算法模块（FLARE 网络、HDR 处理、亚像素定位、测量等） |
| `training/` | 模型训练与评估脚本（train.py、evaluate.py） |
| `data/` | 数据增强与合成数据集生成代码 |
| `docs/` | 技术文档（专利、硬件方案、ABB 集成指南等） |
| `embedded/` | 嵌入式端代码（RA8P1 Helium 优化） |
| `robot/` | ABB RobotStudio 接口与 RAPID 程序 |
| `scripts/` | 数据集预处理脚本（可视化、格式转换等） |
| `tests/` | 单元测试 |
| `dataset/` | 训练数据集（500 组带标注图像） |
| `checkpoints/` | 模型权重文件（best.pth、last.pth、yolo_best.pt） |
| `eval_results/` | 模型评估可视化结果 |
| `main_pipeline.py` | 完整流水线主入口 |
| `demo_streamlit.py` | Streamlit 演示界面 |
| `demo_gradio.py` | Gradio 演示界面 |

**常用命令**：
```bash
# 进入项目目录
cd Anti-Interference-2D-Vision

# 运行主流程（演示模式）
python main_pipeline.py --mode demo --model_path ./checkpoints/best.pth

# 启动 Web 演示
python -m streamlit run demo_streamlit.py --server.port 8501

# 训练模型
python training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8

# 评估模型
python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged
```

---

### 2. assets/（数据资产）
**用途**：存放原始测试数据，不参与版本控制。

| 子目录 | 说明 |
|--------|------|
| `test_samples/` | 10 张标准测试样本 + 1 段测试视频（flare_test_video.mp4） |
| `chesi/` | 5 张额外测试图片（微信来源） |

---

### 3. results/（运行结果）
**用途**：存放算法运行产生的中间结果和日志。

| 子目录 | 说明 |
|--------|------|
| `auto_tuning/` | FLARE 自动优化引擎的 140 次迭代产物（原 ITERATION_COMPOUND/） |

> 运行 `scripts/auto_optimize.py` 会自动向此目录写入新的优化结果。

---

### 4. scripts/（根目录级脚本）
**用途**：一键运行脚本，从根目录直接调用项目代码。

| 文件 | 说明 |
|------|------|
| `auto_optimize.py` | FLARE 全自动 1000 次迭代算法优化引擎 |
| `run_evaluation.bat` | 运行模型评估 |
| `start_demo.bat` | 启动 Streamlit 演示服务器 |
| `start_training.bat` | 启动模型训练（50 epochs） |
| `train_and_demo.bat` | 交互式训练/演示/评估菜单 |

---

### 5. competition/（比赛资料）
**用途**：存放比赛相关的全部文档。

| 子目录 | 说明 |
|--------|------|
| `scheme/` | 作品方案书（多个版本：标准版、详细版、算法核心版） |
| `presentation/` | 路演 PPT（FLARE 项目版、22页专业版、30页苹果风格版、8分钟路演版等） |
| `rules/` | 比赛规则、方案模板 |
| `review/` | 专家评审资料（数据集样本、评估指标、修正方案） |

详见 `competition/README.md` 内的完整文档索引。

---

### 6. team/（团队资料）
**用途**：团队编号相关提交材料。

| 子目录 | 说明 |
|--------|------|
| `inovance030853/` | 团队 "马铃薯炒土豆" 提交资料（方案书 + PPT） |

---

### 7. archive/（归档）
**用途**：存放历史版本和废弃代码，保留备查。

| 子目录 | 说明 |
|--------|------|
| `Anti-ference-2D/` | 旧版项目完整副本（含 vision/existence_checking.py 等早期文件） |

---

## 快速开始

### 环境准备
```bash
cd Anti-Interference-2D-Vision
pip install -r requirements.txt
```

### 运行演示
```bash
# 方式一：直接运行脚本
cd scripts
start_demo.bat

# 方式二：进入项目目录手动运行
cd Anti-Interference-2D-Vision
python -m streamlit run demo_streamlit.py --server.port 8501
```

### 运行自动优化
```bash
cd scripts
python auto_optimize.py
# 结果将输出到 ../results/auto_tuning/
```

---

## 注意事项

1. **GitHub 仓库**：仅关联 `Anti-Interference-2D-Vision/` 目录，根目录其他文件夹（assets、results、competition 等）不纳入版本控制。
2. **模型权重**：`checkpoints/` 已配置 `.gitignore` 忽略 *.pt / *.pth 文件，如需同步权重请使用 Git LFS 或网盘。
3. **数据集**：`dataset/` 含 500 张训练图，体积较大，已配置 `.gitignore` 忽略图片文件。
4. **虚拟环境**：项目使用 `.venv/`（位于 `Anti-Interference-2D-Vision/.venv/`），激活后即可运行全部代码。
