# image_motion_detector.py 文档

## 功能概述

`image_motion_detector.py` 是一个图像移动方向检测器，通过比较两张连续图像，检测画面移动并转换为**人物移动方向**。适用于游戏自动化中判断角色移动方向。

## 核心类：MotionDetector

### 初始化

```python
from image_motion_detector import MotionDetector

# 使用默认参数
detector = MotionDetector()

# 自定义参数
detector = MotionDetector(
    region_size=0.15,      # 区域大小比例
    search_range=40,       # 搜索范围（像素）
    use_airtest=True       # 使用AirTest模板匹配
)
```

**参数说明：**
- `region_size`: 检测区域大小比例（相对于图像短边，默认 0.15）
- `search_range`: 模板匹配搜索范围（像素，默认 40）
- `custom_regions`: 自定义区域列表（可选）
- `use_airtest`: 是否使用 AirTest 的 Template 匹配算法（默认 True，更准确）

### 坐标系统

- **dim0**: 垂直方向（从上到下）
- **dim1**: 水平方向（从左到右）

**重要**：返回值是**人物移动方向**（已取反转换）：
- 内部计算：`background_motion = 图2位置 - 图1位置`
- 返回值：`hero_motion = -background_motion`

### 主要方法

#### 1. detector(img1, img2, region, figpath=None)
**便捷检测函数**（推荐使用）

```python
# 定义检测区域
region = [
    [0.128, 0.367, 0.302, 0.543],  # [dim0_start, dim1_start, dim0_end, dim1_end]
    [0.231, 0.568, 0.413, 0.744],
    [0.347, 0.325, 0.647, 0.419],
    [0.647, 0.285, 0.827, 0.475]
]

# 使用文件路径
result = detector.detector("img1.png", "img2.png", region, "visualization.png")

# 使用numpy数组
import cv2
img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.png")
result = detector.detector(img1, img2, region)

# 打印结果
up, down, left, right = result
print(f"上:{up:.3f}, 下:{down:.3f}, 左:{left:.3f}, 右:{right:.3f}")
```

**参数：**
- `img1`: 参考图像（文件路径字符串 或 numpy数组）
- `img2`: 目标图像（文件路径字符串 或 numpy数组）
- `region`: 检测区域列表 `[[dim0_start, dim1_start, dim0_end, dim1_end], ...]`
- `figpath`: 可选，保存可视化图像路径

**返回值：**
```python
[上, 下, 左, 右]  # 四个浮点数（0-1范围）

# 计算规则：
# 上 = abs(f_dim0) if f_dim0 < 0 else 0
# 下 = abs(f_dim0) if f_dim0 > 0 else 0
# 左 = abs(f_dim1) if f_dim1 < 0 else 0
# 右 = abs(f_dim1) if f_dim1 > 0 else 0
```

**示例输出：**
- `[0.0, 0.05, 0.0, 0.03]` → 人物向下 0.05，向右 0.03
- `[0.02, 0.0, 0.04, 0.0]` → 人物向上 0.02，向左 0.04

#### 2. detect_motion(image1, image2)
**详细检测函数**

```python
result = detector.detect_motion(image1, image2)

print(f"方向: {result['direction']}")      # UP/DOWN/LEFT/RIGHT/NONE
print(f"dim0: {result['f_dim0']:.4f}")     # 垂直方向移动
print(f"dim1: {result['f_dim1']:.4f}")     # 水平方向移动
print(f"置信度: {result['confidence']:.2f}")
```

**返回值：**
```python
{
    'f_dim1': float,       # dim1方向人物移动比例
    'f_dim0': float,       # dim0方向人物移动比例
    'direction': str,      # 主方向：UP/DOWN/LEFT/RIGHT/NONE
    'confidence': float,   # 综合置信度（0-1）
    'valid': bool          # 是否有效
}
```

#### 3. add_custom_rectangular_regions(rectangles)
添加自定义矩形检测区域

```python
rectangles = [
    [0.1, 0.2, 0.3, 0.4],  # [dim0_start, dim1_start, dim0_end, dim1_end]
    [0.5, 0.6, 0.7, 0.8]
]
detector.add_custom_rectangular_regions(rectangles)
```

#### 4. visualize_motion_on_image(image, f_dim1, f_dim0, regions_info, save_path)
可视化移动方向（绘制箭头和区域框）

```python
detector.visualize_motion_on_image(
    image,
    f_dim1=0.02,
    f_dim0=-0.03,
    regions_info=None,
    save_path="motion_vis.png"
)
```

## 检测原理

### 工作流程

1. **区域提取**：从 image1 中提取多个检测区域
2. **模板匹配**：在 image2 中搜索对应区域的新位置
3. **偏移计算**：计算背景移动向量 `(dx, dy)`
4. **方向转换**：��反得到人物移动 `hero_motion = -background_motion`
5. **加权平均**：多区域结果加权平均（按置信度）

### 匹配算法

支持两种匹配算法（`use_airtest` 参数控制）：

1. **AirTest Template 匹配**（推荐，默认）
   - 更准确，基于特征点匹配
   - 计算两次匹配置信度的乘积

2. **相关系数匹配**（备用）
   - 基于图像相关性计算
   - 速度稍快但准确度较低

## 使用示例

### 基础使用

```python
from image_motion_detector import MotionDetector
import cv2

# 1. 创建检测器
detector = MotionDetector()

# 2. 定义检测区域（王者荣耀最佳区域）
region = [
    [0.128, 0.367, 0.302, 0.543],
    [0.231, 0.568, 0.413, 0.744],
    [0.347, 0.325, 0.647, 0.419],
    [0.647, 0.285, 0.827, 0.475]
]

# 3. 检测移动方向
result = detector.detector("frame1.png", "frame2.png", region)
up, down, left, right = result

# 4. 判断主方向
if down > 0.01:
    print(f"向下移动: {down:.3f}")
elif up > 0.01:
    print(f"向上移动: {up:.3f}")
elif right > 0.01:
    print(f"向右移动: {right:.3f}")
elif left > 0.01:
    print(f"向左移动: {left:.3f}")
else:
    print("未移动")
```

### 使用详细结果

```python
detector.add_custom_rectangular_regions(region)

result = detector.detect_motion(img1, img2)

if result['valid']:
    print(f"检测到移动: {result['direction']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"dim0移动: {result['f_dim0']:+.4f}")
    print(f"dim1移动: {result['f_dim1']:+.4f}")
else:
    print("未检测到有效移动")
```

### 批量处理视频帧

```python
import os

detector = MotionDetector()
region = [[0.128, 0.367, 0.302, 0.543], ...]  # 检测区域

# 获取所有帧
frames = sorted([f for f in os.listdir("frames") if f.endswith(".png")])

# 逐帧检测
for i in range(len(frames) - 1):
    img1_path = os.path.join("frames", frames[i])
    img2_path = os.path.join("frames", frames[i+1])

    result = detector.detector(img1_path, img2_path, region)
    up, down, left, right = result

    # 确定主方向
    max_val = max(up, down, left, right)
    if max_val > 0.005:  # 阈值过滤
        direction = ["UP", "DOWN", "LEFT", "RIGHT"][[up, down, left, right].index(max_val)]
        print(f"Frame {i}->{i+1}: {direction} ({max_val:.3f})")
```

### 实时可视化

```python
detector.add_custom_rectangular_regions(region)

for i in range(len(frames) - 1):
    img1 = cv2.imread(frames[i])
    img2 = cv2.imread(frames[i+1])

    motion_result = detector.detect_motion(img1, img2)

    # 生成可视化（第一帧标注区域和箭头）
    if i == 0:
        vis_image = detector.visualize_motion_on_image(
            img1,
            motion_result['f_dim1'],
            motion_result['f_dim0'],
            save_path=f"vis_{i:03d}.png"
        )
        cv2.imshow("Motion", vis_image)
        cv2.waitKey(500)
```

## 王者荣耀最佳配置

### 推荐检测区域

```python
# 经过测试的最佳区域（避开UI元素，选择场景特征丰富区域）
wzry_region = [
    [0.128, 0.367, 0.302, 0.543],  # 区域1：左侧地图区域
    [0.231, 0.568, 0.413, 0.744],  # 区域2：右下场景
    [0.347, 0.325, 0.647, 0.419],  # 区域3：中上场景
    [0.647, 0.285, 0.827, 0.475]   # 区域4：右上场景
]
```

### 推荐参数

```python
detector = MotionDetector(
    region_size=0.15,      # 区域大小（图像短边的15%）
    search_range=40,       # 搜索范围40像素
    use_airtest=True       # 使用AirTest匹配
)
```

### 如何选择检测区域

**选择原则：**
1. ✓ **特征丰富**：建筑、地形、草丛等（易于匹配）
2. ✓ **相对静止**：背景元素，不是英雄/小兵
3. ✗ **避开UI**：技能按钮、血条、小地图
4. ✗ **避开动态元素**：技能特效、伤害数字

**选择步骤：**

1. **截取连续帧**
   ```python
   # 使用 image_capture.py 截取2-3张连续图像
   ```

2. **观察差异**
   - 打开图像查看工具
   - 找到移动时位置变化明显的静态元素

3. **确定坐标**
   ```python
   # 获取像素坐标 (x1, y1, x2, y2)
   # 转换为相对坐标：
   dim0_start = y1 / image_height
   dim1_start = x1 / image_width
   dim0_end = y2 / image_height
   dim1_end = x2 / image_width
   ```

4. **测试验证**
   ```python
   # 使用 main() 函数测试
   python image_motion_detector.py --image_folder test_frames
   ```

5. **调整优化**
   - 检查可视化结果（motion_vis.png）
   - 查看各区域置信度（绿色=高，红色=低）
   - 移除低置信度区域，添加新区域

### 如何调整搜索范围

**`search_range` 参数影响：**
- **太小**（<20）：可能找不到移动后的位置
- **太大**（>60）：计算慢，可能匹配错误位置
- **推荐**：30-50像素（根据移动速度调整）

**调整建议：**
```python
# 慢速移动场景（走路）
detector = MotionDetector(search_range=30)

# 中速移动��跑步）
detector = MotionDetector(search_range=40)

# 快速移动（闪现、冲刺）
detector = MotionDetector(search_range=60)
```

## 命令行工具

### 批量检测图像对

```bash
# 使用默认参数检测 pic_data 文件夹
python image_motion_detector.py

# 指定文件夹
python image_motion_detector.py --image_folder my_frames

# 自定义参数
python image_motion_detector.py \
    --image_folder frames \
    --region_size 0.2 \
    --search_range 50

# 禁用 AirTest 匹配（使用传统方法）
python image_motion_detector.py --no_airtest
```

### 输出说明

```
=== Image Motion Direction Detection ===
Found 10 images

Reference image shape: [1080,1920,3] = (dim0, dim1, RGB)
坐标系: dim0=垂直方向(从上到下), dim1=水平方向(从左到右)
返回值: f_dim0, f_dim1 = 人物移动方向（背景移动的相反方向）

frame_001.png -> frame_002.png: f_dim0=-0.0123(+13.3px), f_dim1=0.0085(-16.3px)
    人物移动: UP, 背景移动: [DOWN, LEFT], confidence=0.875

frame_002.png -> frame_003.png: f_dim0=0.0050, f_dim1=-0.0032, DOWN
...

Motion visualization saved: motion_vis.png
```

## 依赖项

- **opencv-python** (cv2): 图像处理
- **numpy**: 数组计算
- **airtest** (可选): Template 匹配算法（推荐安装）

## 注意事项

1. **图像质量**：清晰度影响匹配准确度
2. **帧间隔**：间隔过大可能超出搜索范围
3. **场景变化**：技能特效、过场动画会影响检测
4. **分辨率**：支持任意分辨率（使用相对坐标）
5. **性能**：AirTest 匹配比传统方法慢约 2-3 倍，但更准确

## 常见问题

### Q: 返回值都是 0 怎么办？
A:
1. 检查两张图像是否有移动
2. 增大 `search_range`
3. 调整检测区域（选择特征更明显的区域）
4. 查看 `confidence`，低于 0.5 说明匹配失败

### Q: 方向判断相反？
A: 本库返回的是**人物移动方向**（已取反），不是背景移动方向。如需背景方向，对返回值取反即可。

### Q: 如何提高检测速度？
A:
1. 减少检测区域数量（2-3个足够）
2. 减小 `region_size`
3. 使用 `use_airtest=False`（牺牲准确度）

### Q: 多个区域结果不一致？
A: 正常现象。系统会自动加权平均（高置信度区域权重更大）。可以查看可视化结果，移除不稳定的区域。

### Q: 支持视频输入吗？
A: 不直接支持。需要先用 OpenCV 提取帧，然后逐帧检测：
```python
import cv2

cap = cv2.VideoCapture("video.mp4")
ret, prev_frame = cap.read()

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    result = detector.detector(prev_frame, curr_frame, region)
    # 处理结果...

    prev_frame = curr_frame
```

## 高级技巧

### 自适应阈值

```python
# 根据移动幅度动态判断
result = detector.detector(img1, img2, region)
up, down, left, right = result

threshold = 0.005  # 最小移动阈值

if max(result) > threshold:
    # 有效移动
    direction = ["UP", "DOWN", "LEFT", "RIGHT"][result.index(max(result))]
else:
    # 静止不动
    direction = "NONE"
```

### 方向平滑

```python
from collections import deque

# 使用滑动窗口平滑方向判断
history = deque(maxlen=5)

for frame_pair in frame_pairs:
    result = detector.detector(*frame_pair, region)
    history.append(result)

    # 平均最近5帧的结果
    avg_result = [sum(x)/len(history) for x in zip(*history)]
    up, down, left, right = avg_result
    # 使用 avg_result 做判断...
```
