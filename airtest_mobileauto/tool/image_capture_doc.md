# image_capture.py 文档

## 功能概述

`image_capture.py` 是一个基于 AirTest 框架的截图工具库，提供了便捷的设备连接和截图功能。

## 核心类：ImageCapture

### 初始化

```python
from image_capture import ImageCapture

# 使用配置文件初始化
capture_tool = ImageCapture(airtest_config="config.yaml", save_folder="screenshots")
```

**参数说明：**
- `airtest_config`: AirTest 配置文件路径（YAML格式）
- `save_folder`: 截图保存文件夹路径（默认: "pic_data"）

### 主要方法

#### 1. airtest_init()
初始化 AirTest 连接

```python
capture_tool.airtest_init()
```

从配置文件中读取设备连接信息并建立连接。

#### 2. screenshot_airtest(save_path=None)
截取当前屏幕

```python
# 截图但不保存
image = capture_tool.screenshot_airtest()

# 截图并保存到指定路径
image = capture_tool.screenshot_airtest("screenshot_001.png")
```

**参数：**
- `save_path`: 可选，保存路径。如果不指定，则不保存文件

**返回值：**
- `numpy.ndarray`: OpenCV 图像数组（BGR格式），失败返回 None

## 配置文件格式

配置文件为 YAML 格式，需要包含以下字段：

```yaml
# config.yaml 示例
LINK: "Android:///"  # 设备连接字符串
# 或者其他连接方式:
# LINK: "Android://127.0.0.1:5037/emulator-5554"  # 模拟器
# LINK: "iOS:///http://localhost:8100"  # iOS设备
```

## 使用示例

### 基础使用

```python
from image_capture import ImageCapture
import cv2

# 1. 创建截图工具
capture = ImageCapture("config.yaml", "my_screenshots")

# 2. 初始化连接
capture.airtest_init()

# 3. 截图
image = capture.screenshot_airtest()

if image is not None:
    # 处理图像
    print(f"截图尺寸: {image.shape}")

    # 保存图像
    cv2.imwrite("my_screenshot.png", image)
```

### 批量截图

```python
import time

capture = ImageCapture("config.yaml", "batch_screenshots")
capture.airtest_init()

# 每秒截图一次，共截10张
for i in range(10):
    image = capture.screenshot_airtest(f"screenshot_{i:03d}.png")
    print(f"已截取第 {i+1} 张图片")
    time.sleep(1)
```

## 依赖项

- **airtest**: 核心截图和设备控制框架
- **opencv-python** (cv2): 图像处理
- **pyyaml**: 配置文件解析
- **numpy**: 数组处理

## 注意事项

1. **设备连接**：使用前必须先调用 `airtest_init()` 初始化连接
2. **配置文件**：确保 config.yaml 中的 LINK 字段正确配置
3. **权限要求**：
   - Android: 需要 adb 调试权限
   - iOS: 需要安装 tidevice 并授权
4. **错误处理**：如果截图失败，`screenshot_airtest()` 返回 None

## 支持的设备类型

- Android 设备（USB/WiFi）
- Android 模拟器（BlueStacks, LDPlayer, MuMu等）
- iOS 设备（需要 tidevice）
- Docker 容器（redroid）

## 常见问题

### Q: 截图返回 None 怎么办？
A: 检查以下几点：
1. 是否已调用 `airtest_init()`
2. 设备是否正确连接（运行 `adb devices` 检查）
3. 配置文件中的 LINK 是否正确

### Q: 支持多设备同时截图吗？
A: 支持。为每个设备创建独立的 ImageCapture 实例，使用不同的配置文件即可。

### Q: 截图格式是什么？
A: 返回的是 OpenCV 的 numpy 数组（BGR格式），可以直接用于图像处理或使用 `cv2.imwrite()` 保存。
