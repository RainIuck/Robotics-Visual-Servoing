# 环境配置指南

本文档详细说明如何从零开始配置 **IBVS UR5 视觉伺服仿真项目** 所需的完整开发环境。

---

## 一、系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11、macOS 12+、Ubuntu 20.04+ |
| Python | **3.8 ~ 3.11**（推荐 3.10，PyBullet 对 3.12+ 支持不稳定） |
| 内存 | 4GB 以上 |
| 显示 | 需要图形界面（PyBullet GUI 模式） |

---

## 二、安装 Python 环境

### 方案 A：使用 Conda（推荐）

Conda 能更好地管理依赖版本，避免冲突。

**1. 安装 Miniconda**

前往 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) 下载对应系统的安装包并安装。

**2. 创建项目专用虚拟环境**

```bash
conda create -n visual_servoing python=3.10
conda activate visual_servoing
```

**3. 验证 Python 版本**

```bash
python --version
# 应输出: Python 3.10.x
```

---

### 方案 B：使用 venv（轻量替代）

```bash
# 在项目根目录下执行
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# macOS / Linux 激活
source .venv/bin/activate
```

---

## 三、安装项目依赖

激活虚拟环境后，执行以下命令安装所有依赖包：

```bash
pip install pybullet numpy opencv-python matplotlib
```

各包说明：

| 包名 | 版本建议 | 用途 |
|------|----------|------|
| `pybullet` | >= 3.2.5 | 物理仿真引擎，UR5 运动学与雅可比计算 |
| `numpy` | >= 1.23 | 矩阵运算、线性代数（伪逆等） |
| `opencv-python` | >= 4.7 | 图像处理、HSV 分割、特征提取 |
| `matplotlib` | >= 3.6 | 误差收敛曲线绘制 |

也可以直接通过项目根目录的 requirements 文件安装（创建后）：

```bash
pip install -r requirements.txt
```

---

## 四、获取 UR5 URDF 模型文件

PyBullet 默认不包含 UR5 模型，需要单独获取。以下提供两种方式：

### 方式 A：安装 pybullet_robots（最简单）

```bash
pip install pybullet_robots
```

安装后，在代码中通过以下方式获取 UR5 URDF 路径：

```python
import pybullet_robots
import os

# UR5 URDF 路径
urdf_path = os.path.join(
    os.path.dirname(pybullet_robots.__file__),
    "data", "urdf", "ur5", "ur5.urdf"
)
```

---

### 方式 B：手动下载 URDF（备用）

如果方式 A 的 URDF 文件有问题，可以从以下仓库手动下载：

```bash
# 克隆包含 UR5 URDF 的仓库
git clone https://github.com/josepdaniel/UR5Bullet.git

# 将 urdf 文件夹复制到本项目的 urdf/ 目录下
```

或者直接下载单个文件：
- 仓库地址：`https://github.com/josepdaniel/UR5Bullet`
- 需要的文件：`urdf/` 目录下的所有 `.urdf` 和 `.obj` 文件

将下载的文件放入本项目的 `urdf/` 目录：

```
Visual Servoing/
└── urdf/
    ├── ur5.urdf
    └── meshes/       ← 网格文件，必须一起复制
```

---

## 五、验证安装

### 5.1 验证 PyBullet

新建一个测试脚本 `test_pybullet.py`，运行以下代码：

```python
import pybullet as p
import pybullet_data
import time

# 启动 GUI 模式
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加载地面
p.loadURDF("plane.urdf")

# 加载一个测试机器人（Kuka）
robot = p.loadURDF("kuka/kuka.urdf", basePosition=[0, 0, 0])

print(f"机器人关节数: {p.getNumJoints(robot)}")

# 运行 3 秒
for _ in range(240 * 3):
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
print("PyBullet 测试通过！")
```

```bash
python test_pybullet.py
```

**预期结果：** 弹出 PyBullet GUI 窗口，显示机械臂模型，终端输出"PyBullet 测试通过！"。

---

### 5.2 验证 OpenCV

```python
import cv2
import numpy as np

# 创建一张测试图像（红色方块）
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (280, 200), (360, 280), (0, 0, 255), -1)

# HSV 分割测试
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))

# 计算质心
M = cv2.moments(mask)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    print(f"检测到质心: ({cx}, {cy})")
    print("OpenCV 测试通过！")
```

```bash
python test_opencv.py
```

**预期结果：** 终端输出质心坐标（应接近 320, 240）。

---

### 5.3 验证 NumPy 矩阵运算

```python
import numpy as np

# 模拟交互矩阵伪逆计算
u, v, Z = 0.1, 0.05, 0.5
L_s = np.array([
    [-1/Z,  0,    u/Z,  u*v,       -(1+u**2),  v],
    [ 0,   -1/Z,  v/Z,  1+v**2,   -u*v,       -u]
])

L_s_pinv = np.linalg.pinv(L_s)
print(f"交互矩阵形状: {L_s.shape}")
print(f"伪逆形状: {L_s_pinv.shape}")
print("NumPy 测试通过！")
```

**预期结果：** 输出 `(2, 6)` 和 `(6, 2)`。

---

## 六、IDE 配置建议

推荐使用 **VS Code**，安装以下扩展：

- `Python`（Microsoft 官方）
- `Pylance`（类型检查与自动补全）

配置 Python 解释器：
1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. 选择刚才创建的 `visual_servoing` conda 环境

---

## 七、常见问题

### Q1：PyBullet 安装失败（Windows）

```
error: Microsoft Visual C++ 14.0 is required
```

**解决方案：** 安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，或改用预编译版本：

```bash
pip install pybullet --only-binary=:all:
```

---

### Q2：PyBullet GUI 窗口无法显示（远程服务器）

将 GUI 模式改为 DIRECT 模式（无图形界面）：

```python
# 将 p.connect(p.GUI) 改为
physicsClient = p.connect(p.DIRECT)
```

---

### Q3：OpenCV `imshow` 在某些环境下卡死

```bash
# 改用 headless 版本
pip uninstall opencv-python
pip install opencv-python-headless
```

---

### Q4：找不到 UR5 URDF 文件

确认 `urdf/` 目录结构正确，并在代码中使用绝对路径：

```python
import os
URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "ur5.urdf")
```

---

### Q5：`pybullet_robots` 中 UR5 路径找不到

```python
# 用以下方式定位实际安装路径
import pybullet_robots
print(pybullet_robots.__file__)
# 在该目录下手动查找 ur5.urdf
```

---

## 八、快速检查清单

完成配置后，逐项确认：

- [ ] Python 3.10 虚拟环境已激活
- [ ] `pip list` 中能看到 `pybullet`、`opencv-python`、`numpy`、`matplotlib`
- [ ] `test_pybullet.py` 运行成功，GUI 窗口正常弹出
- [ ] `test_opencv.py` 运行成功，质心坐标输出正确
- [ ] UR5 URDF 文件已放入 `urdf/` 目录（或通过 `pybullet_robots` 可访问）

全部勾选后，环境配置完成，可以开始编写项目代码。
