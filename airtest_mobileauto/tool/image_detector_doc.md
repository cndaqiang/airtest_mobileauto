# image_detector.py 文档

## 功能概述

`image_detector.py` 是一个通用图像状态检测器，通过分析图像区域的亮度、对比度、饱和度等特征来判断状态（如游戏角色存活/死亡）。

## 核心类：ImageDetector

### 初始化

```python
from image_detector import ImageDetector

# 使用默认参数（右下角按钮区域，阈值80）
detector = ImageDetector()

# 自定义区域和阈值
regions = [
    {"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50},
    {"name": "attack_button", "center_dim0": 0.86, "center_dim1": 0.91}
]
detector = ImageDetector(regions=regions, threshold=75)
```

**参数说明：**
- `regions`: 检测区域列表（默认检测右下角 90%, 50%）
- `threshold`: 分数阈值，默认 80（范围 0-100）

### 坐标系统

本库使用 dim0/dim1 坐标系统：
- **dim0**: 垂直方向（从上到下，0.0=顶部，1.0=底部）
- **dim1**: 水平方向（从左到右，0.0=左侧，1.0=右侧）

例如：`center_dim0=0.9, center_dim1=0.5` 表示屏幕下方中央位置。

### 主要方法

#### 1. detect(image, visualize=False, save_path=None)
检测图像状态

```python
import cv2

image = cv2.imread("screenshot.png")
result = detector.detect(image)

print(f"状态: {'存活' if result['is_alive'] else '死亡'}")
print(f"综合分数: {result['combined_score']:.1f}")
```

**参数：**
- `image`: 输入图像（numpy数组或文件路径字符串）
- `visualize`: 是否可视化（默认 False）
- `save_path`: 可视化结果保存路径（可选）

**返回值：**
```python
{
    'is_alive': bool,           # 是否存活
    'combined_score': float,    # 综合分数（0-100）
    'regions': [                # 各区域详细信息
        {
            'name': str,        # 区域名称
            'score': float,     # 区域分数
            'is_alive': bool    # 区域判定结果
        },
        ...
    ]
}
```

#### 2. set_threshold(threshold)
设置判定阈值

```python
detector.set_threshold(75)  # 设置阈值为75
```

#### 3. add_region(name, center_dim0, center_dim1)
添加检测区域

```python
detector.add_region("custom_area", 0.8, 0.3)
```

#### 4. clear_regions()
清空所有检测区域

```python
detector.clear_regions()
```

## 检测原理

检测器通过分析图像区域的多个特征来判断状态：

### 特征权重（可自定义）

```python
detector.algorithm_weights = {
    'brightness': 0.4,      # 亮度权重（40%）
    'contrast': 0.2,        # 对比度权重（20%）
    'saturation': 0.25,     # 饱和度权重（25%）
    'color_variance': 0.15  # 色彩差异权重（15%）
}
```

### 判定逻辑

1. 提取每个检测区域的图像块
2. 计算各项特征分数（0-100）
3. 加权平均得到区域综合分数
4. 与阈值比较：`分数 >= 阈值` → 存活，否则死亡
5. 多区域时取平均分数

## 使用示例

### 基础检测

```python
from image_detector import ImageDetector
import cv2

# 1. 创建检测器（王者荣耀技能按钮区域）
regions = [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]
detector = ImageDetector(regions=regions, threshold=80)

# 2. 检测图像
image = cv2.imread("screenshot.png")
result = detector.detect(image)

# 3. 查看结果
if result['is_alive']:
    print(f"✓ 存活 (分数: {result['combined_score']:.1f})")
else:
    print(f"✗ 死亡 (分数: {result['combined_score']:.1f})")
```

### 批量检测并分类

```python
import os
import shutil

detector = ImageDetector(threshold=80)

# 创建分类文件夹
os.makedirs("alive", exist_ok=True)
os.makedirs("dead", exist_ok=True)

# 批量检测
for filename in os.listdir("screenshots"):
    if not filename.endswith(".png"):
        continue

    image_path = os.path.join("screenshots", filename)
    result = detector.detect(image_path)

    # 根据检测结果分类
    if result['is_alive']:
        shutil.copy(image_path, f"alive/{filename}")
    else:
        shutil.copy(image_path, f"dead/{filename}")

    print(f"{filename}: {'存活' if result['is_alive'] else '死亡'} ({result['combined_score']:.1f})")
```

### 可视化检测区域

```python
# 检测并保存可视化结果（标注检测区域）
result = detector.detect(
    "screenshot.png",
    visualize=True,
    save_path="detection_result.png"
)
```

### 阈值调优

```python
# 扫描不同阈值，找到最佳值
for threshold in range(60, 101, 5):
    detector.set_threshold(threshold)

    alive_correct = 0
    dead_correct = 0

    # 测试存活样本
    for img_path in alive_images:
        result = detector.detect(img_path)
        if result['is_alive']:
            alive_correct += 1

    # 测试死亡样本
    for img_path in dead_images:
        result = detector.detect(img_path)
        if not result['is_alive']:
            dead_correct += 1

    accuracy = (alive_correct + dead_correct) / (len(alive_images) + len(dead_images))
    print(f"阈值 {threshold}: 准确率 {accuracy*100:.1f}%")
```

## 王者荣耀最佳配置

### 推荐区域

```python
# 方案1：仅技能区域（推荐）
regions = [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]

# 方案2：技能+普攻双区域（更准确）
regions = [
    {"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50},
    {"name": "attack_button", "center_dim0": 0.86, "center_dim1": 0.91}
]
```

### 推荐阈值

- **技能区域单独使用**: 75-80
- **双区域使用**: 70-75
- **建议**: 使用 demo_deatch_detector.py 的阈值调优功能找到最佳值

### 如何选择区域

1. **截取测试图像**：存活和死亡状态各若干张
2. **观察差异**：找到存活时亮、死亡时暗的区域
3. **确定坐标**：
   - 使用图像查看工具获取像素坐标 (x, y)
   - 转换为相对坐标：`dim0 = y / 图像高度`, `dim1 = x / 图像宽度`
4. **测试验证**：使用 demo_deatch_detector.py 验证效果

### 如何调整阈值

1. **初始值**: 从 80 开始
2. **误判情况**：
   - 存活误判为死亡 → 降低阈值（如 75, 70）
   - 死亡误判为存活 → 提高阈值（如 85, 90）
3. **使用工具**: `demo_deatch_detector.py` 提供自动阈值扫描功能

## 依赖项

- **opencv-python** (cv2): 图像处理
- **numpy**: 数组计算

## 注意事项

1. **图像格式**: 支持 BGR 和 RGB，内部会自动转换
2. **区域大小**: 默认为图像短边的 3%，可通过 `size_ratio` 参数调整
3. **光照影响**: 检测结果可能受屏幕亮度影响，建议固定亮度
4. **分辨率**: 适用于各种分辨率，坐标使用相对值（0-1）

## 常见问题

### Q: 为什么检测不准？
A:
1. 检查检测区域是否正确（是否在按钮位置）
2. 尝试调整阈值
3. 确保测试图像与实际场景一致（分辨率、UI布局）

### Q: 如何处理不同分辨率？
A: 使用相对坐标（0-1范围），自动适配不同分辨率

### Q: 多区域如何判定？
A: 取所有区域的平均分数与阈值比较

### Q: 可以检测其他状态吗？
A: 可以！只需找到特征明显的区域，调整阈值即可用于各种二分类判断（开/关、有/无等）
