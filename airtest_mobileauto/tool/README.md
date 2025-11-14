# AirTest Mobile Auto - Tool 模块

基于 AirTest 框架的移动设备自动化工具集，提供截图、状态检测和移动方向识别功能。

## 📁 项目结构

```
tool/
├── image_capture.py              # 核心库：截图工具
├── image_detector.py             # 核心库：图像状态检测器
├── image_motion_detector.py      # 核心库：移动方向检测器
├── demo_deatch_detector.py       # 示例：死亡检测和分类（王者荣耀）
├── demo_capture_and_move_detector.py  # 示例：实时截图和移动检测（王者荣耀）
├── image_capture_doc.md          # image_capture 文档
├── image_detector_doc.md         # image_detector 文档
├── image_motion_detector_doc.md  # image_motion_detector 文档
└── README.md                     # 本文件
```

## 🚀 快速开始

### 环境要求

```bash
# 安装依赖
pip install opencv-python numpy pyyaml airtest
```

### 基础工作流程

#### 1️⃣ 配置设备连接

创建 `config.yaml` 文件：

```yaml
# Android 设备
LINK: "Android:///"

# 或 Android 模拟器
# LINK: "Android://127.0.0.1:5037/emulator-5554"

# 或 iOS 设备
# LINK: "iOS:///http://localhost:8100"
```

#### 2️⃣ 截取测试图像

```python
from image_capture import ImageCapture

# 初始化截图工具
capture = ImageCapture("config.yaml", "screenshots")
capture.airtest_init()

# 截图
image = capture.screenshot_airtest("test.png")
```

或使用命令行（如果 demo 支持）：
```bash
python demo_capture_and_move_detector.py --config config.yaml
```

#### 3️⃣ 状态检测（死亡/存活判定）

```python
from image_detector import ImageDetector

# 创建检测器（王者荣耀技能按钮区域）
regions = [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]
detector = ImageDetector(regions=regions, threshold=80)

# 检测图像
result = detector.detect("screenshot.png")
print(f"状态: {'存活' if result['is_alive'] else '死亡'}")
```

或使用 demo 工具：
```bash
python demo_deatch_detector.py
```

#### 4️⃣ 移动方向检测

```python
from image_motion_detector import MotionDetector

# 创建检测器
detector = MotionDetector()

# 定义检测区域（王者荣耀最佳区域）
region = [
    [0.128, 0.367, 0.302, 0.543],
    [0.231, 0.568, 0.413, 0.744],
    [0.347, 0.325, 0.647, 0.419],
    [0.647, 0.285, 0.827, 0.475]
]

# 检测移动 [上, 下, 左, 右]
result = detector.detector("frame1.png", "frame2.png", region)
up, down, left, right = result
print(f"移动方向: 上{up:.3f}, 下{down:.3f}, 左{left:.3f}, 右{right:.3f}")
```

或使用命令行工具：
```bash
python image_motion_detector.py --image_folder frames
```

## 📚 核心库详细文档

### 1. image_capture.py - 截图工具

**功能**：基于 AirTest 的设备截图

**核心方法**：
- `airtest_init()`: 初始化设备连接
- `screenshot_airtest(save_path)`: 截图

**详细文档**: [image_capture_doc.md](image_capture_doc.md)

**示例**：
```python
from image_capture import ImageCapture

capture = ImageCapture("config.yaml", "screenshots")
capture.airtest_init()
image = capture.screenshot_airtest("test.png")  # 返回 numpy 数组
```

---

### 2. image_detector.py - 状态检测器

**功能**：通过分析图像区域特征（亮度、对比度、饱和度）判断状态

**核心方法**：
- `detect(image)`: 检测图像状态
- `set_threshold(threshold)`: 设置判定阈值
- `add_region(name, center_dim0, center_dim1)`: 添加检测区域

**详细文档**: [image_detector_doc.md](image_detector_doc.md)

**示例**：
```python
from image_detector import ImageDetector

detector = ImageDetector(threshold=80)
result = detector.detect("screenshot.png")

if result['is_alive']:
    print(f"✓ 存活 (分数: {result['combined_score']:.1f})")
```

---

### 3. image_motion_detector.py - 移动方向检测器

**功能**：比较两张连续图像，检测人物移动方向

**核心方法**：
- `detector(img1, img2, region, figpath)`: 便捷检测，返回 [上,下,左,右]
- `detect_motion(image1, image2)`: 详细检测，返回方向和置信度
- `add_custom_rectangular_regions(rectangles)`: 设置检测区域

**详细文档**: [image_motion_detector_doc.md](image_motion_detector_doc.md)

**示例**：
```python
from image_motion_detector import MotionDetector

detector = MotionDetector()
region = [[0.1, 0.2, 0.3, 0.4], ...]

# 返回 [上, 下, 左, 右]
result = detector.detector("img1.png", "img2.png", region)
```

## 🎮 Demo 示例说明

### demo_deatch_detector.py - 死亡检测和分类

**适用场景**：王者荣耀角色存活/死亡判定

**功能**：
1. 批量检测图像并分类（alive/dead 文件夹）
2. 阈值扫描和调优（找到最佳阈值）
3. 可视化检测区域
4. 准确率统计

**使用方法**：
```bash
# 交互式运行
python demo_deatch_detector.py

# 按提示操作：
# 1. 选择检测区域（推荐：技能按钮区域）
# 2. 选择模式（快速判断 或 阈值调优）
# 3. 查看结果和统计
```

**区域配置**（针对王者荣耀）：
- 技能按钮区域：`center_dim0=0.90, center_dim1=0.50`
- 普攻按钮区域：`center_dim0=0.86, center_dim1=0.91`
- 推荐阈值：75-80

---

### demo_capture_and_move_detector.py - 实时移动检测

**适用场景**：王者荣耀实时移动方向识别

**功能**：
1. 实时截图并检测移动方向
2. 每 N 张自动保存图像对
3. 第一张图标注检测区域和箭头
4. 方向统计

**使用方法**：

```bash
# 连续模式（自动截图）
python demo_capture_and_move_detector.py \
    --config config.yaml \
    --save_interval 5 \
    --interval 0.5

# 交互模式（按 Enter 手动截图）
python demo_capture_and_move_detector.py --interactive
```

**参数说明**：
- `--config`: 配置文件路径
- `--save_interval`: 每隔几张保存一次（默认 5）
- `--interval`: 截图间隔秒数（默认 0.5）
- `--max_captures`: 最大截图数量
- `--interactive`: 交互模式

**检测区域**（针对王者荣耀）：
```python
region = [
    [0.128, 0.367, 0.302, 0.543],  # 左侧地图区域
    [0.231, 0.568, 0.413, 0.744],  # 右下场景
    [0.347, 0.325, 0.647, 0.419],  # 中上场景
    [0.647, 0.285, 0.827, 0.475]   # 右上场景
]
```

## 🎯 王者荣耀最佳配置

### 状态检测（死亡判定）

**推荐区域**：
```python
# 方案1：单区域（推荐）
regions = [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]

# 方案2：双区域（更稳定）
regions = [
    {"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50},
    {"name": "attack_button", "center_dim0": 0.86, "center_dim1": 0.91}
]
```

**推荐阈值**：
- 单区域：75-80
- 双区域：70-75

**调优方法**：
```bash
python demo_deatch_detector.py
# 选择 "阈值调优模式"
# 系统会自动扫描 60-100 的阈值，找到最佳值
```

---

### 移动方向检测

**推荐区域**（避开 UI，选择场景元素）：
```python
wzry_region = [
    [0.128, 0.367, 0.302, 0.543],  # 区域1
    [0.231, 0.568, 0.413, 0.744],  # 区域2
    [0.347, 0.325, 0.647, 0.419],  # 区域3
    [0.647, 0.285, 0.827, 0.475]   # 区域4
]
```

**推荐参数**：
```python
detector = MotionDetector(
    region_size=0.15,
    search_range=40,
    use_airtest=True
)
```

## 🛠️ 如何自定义区域和阈值

### 状态检测区域选择

**步骤**：

1. **截取样本**
   ```python
   # 分别截取存活和死亡状态的图像各 10+ 张
   ```

2. **观察差异**
   - 找到存活时**亮**、死亡时**暗**的区域
   - 常见位置：技能按钮、血条、复活按钮

3. **确定坐标**
   ```python
   # 使用图像查看工具获取像素坐标 (x, y)
   # 转换为相对坐标：
   center_dim0 = y / image_height  # 垂直位置
   center_dim1 = x / image_width   # 水平位置
   ```

4. **测试验证**
   ```bash
   python demo_deatch_detector.py
   # 选择 "自定义区域"，输入坐标
   ```

**坐标系说明**：
- `dim0`: 垂直方向，0.0=顶部，1.0=底部
- `dim1`: 水平方向，0.0=左侧，1.0=右侧
- 例如：`(0.9, 0.5)` = 屏幕下方中央

---

### 状态检测阈值调整

**调整原则**：

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 存活误判为死亡 | 阈值过高 | 降低阈值（80→75→70）|
| 死亡误判为存活 | 阈值过低 | 提高阈值（80→85→90）|
| 准确率不稳定 | 区域选择不当 | 更换检测区域 |

**自动调优**：
```bash
python demo_deatch_detector.py
# 选择 "阈值调优模式"
# 输入阈值范围：60-100，步长 5
# 系统会输出各阈值的准确率，选择最佳值
```

---

### 移动检测区域选择

**选择原则**：

✅ **应该选择**：
- 地图背景元素（建筑、地形、草丛）
- 静态场景物体
- 纹理丰富的区域

❌ **避免选择**：
- UI 元素（按钮、血条、小地图）
- 动态元素（英雄、小兵、技能特效）
- 纯色区域

**选择步骤**：

1. **截取连续帧**
   ```bash
   # 移动时截取 2-3 张连续图像
   ```

2. **找到特征区域**
   - 打开图像编辑器（如 Paint, GIMP）
   - 找到移动时位置变化明显的**静态背景**

3. **框选区域**
   ```python
   # 获取矩形框的像素坐标 (x1, y1, x2, y2)
   # 转换为相对坐标：
   region = [
       y1/height,  # dim0_start
       x1/width,   # dim1_start
       y2/height,  # dim0_end
       x2/width    # dim1_end
   ]
   ```

4. **测试验证**
   ```bash
   python image_motion_detector.py --image_folder test_frames
   # 查看生成的 motion_vis.png
   # 绿色框 = 高置信度（好）
   # 红色框 = 低置信度（需调整）
   ```

5. **迭代优化**
   - 移除低置信度区域（红色）
   - 添加更多高质量区域（绿色）
   - 保留 2-4 个区域即可

---

### 移动检测搜索范围调整

**`search_range` 影响**：
- 值太小：找不到移动后的位置
- 值太大：计算慢，可能误匹配

**调整建议**：

| 移动速度 | search_range | 说明 |
|----------|--------------|------|
| 慢速（走路） | 20-30 | 小范围搜索即可 |
| 中速（跑步） | 30-50 | 推荐默认值 40 |
| 快速（闪现） | 50-80 | 大范围搜索 |

```python
# 根据游戏场景调整
detector = MotionDetector(search_range=40)  # 中速
```

## 📊 完整工作流程示例

### 场景：王者荣耀自动战斗

```python
from image_capture import ImageCapture
from image_detector import ImageDetector
from image_motion_detector import MotionDetector
import time

# 1. 初始化工具
capture = ImageCapture("config.yaml")
capture.airtest_init()

death_detector = ImageDetector(
    regions=[{"name": "skill", "center_dim0": 0.90, "center_dim1": 0.50}],
    threshold=80
)

motion_detector = MotionDetector()
motion_region = [
    [0.128, 0.367, 0.302, 0.543],
    [0.231, 0.568, 0.413, 0.744],
    [0.347, 0.325, 0.647, 0.419],
    [0.647, 0.285, 0.827, 0.475]
]

# 2. 主循环
prev_frame = None

while True:
    # 截图
    curr_frame = capture.screenshot_airtest()

    # 检查存活状态
    status = death_detector.detect(curr_frame)

    if not status['is_alive']:
        print("已死亡，等待复活...")
        time.sleep(3)
        continue

    # 检测移动方向
    if prev_frame is not None:
        move = motion_detector.detector(prev_frame, curr_frame, motion_region)
        up, down, left, right = move

        # 根据移动方向执行操作
        if down > 0.01:
            print(f"向下移动 {down:.3f}")
            # 执行游戏操作...
        elif up > 0.01:
            print(f"向上移动 {up:.3f}")
            # 执行游戏操作...
        # ... 其他方向

    prev_frame = curr_frame
    time.sleep(0.5)
```

## 🔧 故障排除

### 截图失败

**症状**：`screenshot_airtest()` 返回 None

**解决方案**：
1. 检查设备连接：`adb devices`
2. 检查配置文件 LINK 是否正确
3. 确保已调用 `airtest_init()`
4. iOS 设备检查 tidevice 是否安装

---

### 状态检测不准

**症状**：死亡/存活判断错误

**解决方案**：
1. 使用阈值调优工具找到最佳阈值
2. 检查检测区域是否正确（可视化查看）
3. 确保样本图像充足（各 10+ 张）
4. 检查屏幕亮度是否一致

---

### 移动检测无效

**症状**：返回值都是 0 或方向错误

**解决方案**：
1. 检查两张图像是否确实有移动
2. 增大 `search_range`（如 60）
3. 检查检测区域：
   - 查看 motion_vis.png
   - 移除红色（低置信度）区域
   - 选择场景特征更明显的区域
4. 确保没有选到 UI 或动态元素

---

### 性能问题

**症状**：检测速度慢

**解决方案**：
1. 减少检测区域数量（2-3 个即可）
2. 移动检测使用 `use_airtest=False`
3. 降低截图频率
4. 减小 `region_size` 和 `search_range`

## 📖 相关资源

- **AirTest 文档**: https://airtest.doc.io.netease.com/
- **OpenCV 文档**: https://docs.opencv.org/
- **项目主页**: https://github.com/cndaqiang/airtest_mobileauto

## 📝 许可证

与 airtest_mobileauto 主项目保持一致

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**最后更新**: 2025-11-14
