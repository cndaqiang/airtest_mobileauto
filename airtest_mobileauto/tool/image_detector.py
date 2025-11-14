import cv2
import numpy as np
import os
from datetime import datetime

class ImageDetector:
    """
    通用图像状态检测器
    支持自定义检测区域和算法

    坐标系统说明：
    - dim0: 垂直方向（从上到下，对应传统的y轴）
    - dim1: 水平方向（从左到右，对应传统的x轴）
    - 坐标格式：{"name": "区域1", "center_dim0": 0.9, "center_dim1": 0.5}
    """

    def __init__(self, regions=None, threshold=80):
        """
        初始化检测器

        参数:
        regions: 检测区域列表，格式：[{"name": "区域1", "center_dim0": 0.9, "center_dim1": 0.5}, ...]
                默认只检测右下角按钮区域
        threshold: 分数阈值，默认80
        """
        self.threshold = threshold

        # 默认检测区域（右下角按钮区域）
        if regions is None:
            self.regions = [
                {"name": "button_region", "center_dim0": 0.90, "center_dim1": 0.50}
            ]
        else:
            self.regions = regions

        # 默认算法参数
        self.algorithm_weights = {
            'brightness': 0.4,      # 亮度权重
            'contrast': 0.2,        # 对比度权重
            'saturation': 0.25,     # 饱和度权重
            'color_variance': 0.15  # 色彩差异权重
        }

    def add_region(self, name, center_dim0, center_dim1):
        """Add new detection region"""
        self.regions.append({
            "name": name,
            "center_dim0": center_dim0,
            "center_dim1": center_dim1
        })

    def clear_regions(self):
        """Clear all detection regions"""
        self.regions = []

    def _calculate_region_bounds(self, image_shape, region_def, size_ratio=0.03):
        """
        Calculate region boundary information

        Parameters:
        image_shape: image shape (dim0_size, dim1_size, ...)
        region_def: region definition {"name": ..., "center_dim0": ..., "center_dim1": ...}
        size_ratio: region size ratio (percentage of short side)

        Returns:
        dict: region boundary information {
            'name': region name,
            'dim1_start': top-left dim1,
            'dim0_start': top-left dim0,
            'dim1_end': bottom-right dim1,
            'dim0_end': bottom-right dim0,
            'center_dim0': center point dim0,
            'center_dim1': center point dim1,
            'size': region size
        }
        """
        dim0_size, dim1_size = image_shape[:2]

        # Calculate region size (based on percentage of short side)
        short_side = min(dim0_size, dim1_size)
        region_size = int(size_ratio * short_side)

        # Calculate center point pixel coordinates
        center_dim1 = int(region_def["center_dim1"] * dim1_size)
        center_dim0 = int(region_def["center_dim0"] * dim0_size)

        # Calculate region boundaries
        dim1_start = max(0, center_dim1 - region_size // 2)
        dim0_start = max(0, center_dim0 - region_size // 2)
        dim1_end = min(dim1_size, dim1_start + region_size)
        dim0_end = min(dim0_size, dim0_start + region_size)

        return {
            'name': region_def['name'],
            'dim1_start': dim1_start,
            'dim0_start': dim0_start,
            'dim1_end': dim1_end,
            'dim0_end': dim0_end,
            'center_dim0': center_dim0,
            'center_dim1': center_dim1,
            'size': region_size
        }

    def get_region_positions(self, image_shape):
        """
        Get detection region positions in image

        Parameters:
        image_shape: image shape (dim0_size, dim1_size, channels)

        Returns:
        list: region position information [{
            'name': region name,
            'dim1_start': top-left dim1,
            'dim0_start': top-left dim0,
            'dim1_end': bottom-right dim1,
            'dim0_end': bottom-right dim0,
            'center_dim0': center point dim0,
            'center_dim1': center point dim1,
            'size': region size
        }, ...]
        """
        positions = []

        for region_def in self.regions:
            bounds = self._calculate_region_bounds(image_shape, region_def)
            positions.append(bounds)

        return positions

    def extract_region(self, image, region_def, size_ratio=0.03):
        """Extract specified region"""
        bounds = self._calculate_region_bounds(image.shape, region_def, size_ratio)
        return image[bounds['dim0_start']:bounds['dim0_end'], bounds['dim1_start']:bounds['dim1_end']]

    def calculate_score(self, image_or_path, algorithm='default'):
        """
        Calculate image score

        Parameters:
        image_or_path: cv2 image array or image file path
        algorithm: algorithm type, currently supports 'default'

        Returns:
        float: calculated score
        """
        # Process input
        if isinstance(image_or_path, str):
            # File path
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_or_path}")
        else:
            # cv2 image array
            image = image_or_path

        if algorithm == 'default':
            return self._calculate_default_score(image)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _calculate_default_score(self, image):
        """
        Default algorithm to calculate score

        Parameters:
        image: cv2 image array

        Returns:
        float: comprehensive score
        """
        total_score = 0
        region_count = 0

        for region_def in self.regions:
            # Extract detection region
            region = self.extract_region(image, region_def)
            if region.size == 0:
                continue

            # Calculate region score
            region_score = self._calculate_region_score(region)
            total_score += region_score
            region_count += 1

        # Return average score
        return total_score / region_count if region_count > 0 else 0

    def _calculate_region_score(self, region):
        """
        Calculate score for a single region

        Parameters:
        region: image region

        Returns:
        float: region score
        """
        # 转换为灰度图
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # 基础亮度
        brightness = np.mean(gray)

        # 对比度
        contrast = np.std(gray)

        # 色彩丰富度
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])

        # RGB色彩差异
        b, g, r = cv2.split(region)
        color_variance = np.std([r, g, b])

        # 综合得分计算
        score = (brightness * self.algorithm_weights['brightness'] +
                contrast * self.algorithm_weights['contrast'] +
                saturation * self.algorithm_weights['saturation'] +
                color_variance * self.algorithm_weights['color_variance'])

        return score

    def detect(self, image_or_path):
        """
        Detect image status

        Parameters:
        image_or_path: cv2 image array or image file path

        Returns:
        tuple: (status, score)
               status: True=high score state, False=low score state
               score: calculated score
        """
        score = self.calculate_score(image_or_path)
        is_high_score = score > self.threshold
        return is_high_score, score

    def set_threshold(self, threshold):
        """Set threshold"""
        self.threshold = threshold

    def get_threshold(self):
        """Get current threshold"""
        return self.threshold

    def visualize_regions(self, image_or_path, save_path=None):
        """
        Visualize detection regions

        Parameters:
        image_or_path: cv2 image array or image file path
        save_path: save path (optional)

        Returns:
        np.ndarray: image with detection regions marked
        """
        # Load image
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_or_path}")
        else:
            image = image_or_path.copy()

        # Get region positions
        positions = self.get_region_positions(image.shape)

        # Draw regions
        for pos in positions:
            # Draw rectangle (cv2 uses (x,y) format, so we map dim1->x, dim0->y)
            cv2.rectangle(image, (pos['dim1_start'], pos['dim0_start']), (pos['dim1_end'], pos['dim0_end']), (0, 255, 0), 2)

            # Add text label
            cv2.putText(image, pos['name'], (pos['dim1_start'], pos['dim0_start'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Calculate and display region score
            region = image[pos['dim0_start']:pos['dim0_end'], pos['dim1_start']:pos['dim1_end']]
            if region.size > 0:
                score = self._calculate_region_score(region)
                cv2.putText(image, f"Score: {score:.1f}", (pos['dim1_start'], pos['dim0_end'] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 显示整体得分
        total_score = self.calculate_score(image)
        status_text = "HIGH" if total_score > self.threshold else "LOW"
        cv2.putText(image, f"Total: {total_score:.1f} ({status_text})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 保存图像
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Visualization saved: {save_path}")

        return image

    def evaluate_folder(self, folder_path, expected_status, verbose=True):
        """
        Evaluate images in folder

        Parameters:
        folder_path: folder path
        expected_status: expected status (True=state A, False=state B)
        verbose: whether to print detailed information

        Returns:
        list: evaluation results
        """
        results = []

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return results

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)

                try:
                    # 计算得分
                    score = self.calculate_score(filepath)
                    # 判断状态
                    predicted_status = score > self.threshold
                    # 是否正确
                    correct = predicted_status == expected_status

                    result = {
                        'filename': filename,
                        'score': score,
                        'predicted_status': predicted_status,
                        'expected_status': expected_status,
                        'correct': correct
                    }
                    results.append(result)

                    if verbose:
                        status_type = "HIGH" if predicted_status else "LOW"
                        mark = "+" if correct else "-"
                        print(f"{mark} {filename}: Score={score:.1f}, Status={status_type}")

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        return results