#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像移动方向检测器
基于simple_motion_detector.py的测试验证算法

功能：检测人物移动方向（返回值已转换为人物移动，而非背景移动）
- 内部计算背景移动：background_motion = 图2位置 - 图1位置
- 返回人物移动：hero_motion = -background_motion
- 箭头显示：人物移动方向
"""

import cv2
import numpy as np
import os
import re
from datetime import datetime

try:
    from airtest.core.cv import Template
    import logging
    logger = logging.getLogger("airtest")
    logger.setLevel(logging.WARNING)
    AIRTEST_AVAILABLE = True
except ImportError:
    AIRTEST_AVAILABLE = False
    print("Warning: airtest not available, falling back to correlation matching")

class MotionDetector:
    """
    图像移动方向检测器
    核心功能：检测两张图像之间的移动方向和幅度
    """

    def __init__(self, region_size=0.15, search_range=40, custom_regions=None, use_airtest=True):
        """
        初始化检测器

        参数:
        region_size: 区域大小比例（相对于图像短边，默认0.15）
        search_range: 搜索范围（像素，默认40）
        custom_regions: 自定义区域列表，格式：[{"name": "区域1", "dim0_start": 0.1, "dim1_start": 0.2, "dim0_end": 0.3, "dim1_end": 0.4}, ...]
        use_airtest: 是否使用airtest的Template匹配算法（默认True）
        """
        self.region_size_ratio = region_size
        self.search_range = search_range
        self.use_airtest = use_airtest and AIRTEST_AVAILABLE

        if custom_regions:
            # 使用自定义区域
            self.regions = custom_regions
        else:
            # 默认检测区域（中心+四角）
            self.regions = [
                {"name": "center", "center_dim0": 0.5, "center_dim1": 0.5},
                {"name": "top_left", "center_dim0": 0.25, "center_dim1": 0.25},
                {"name": "top_right", "center_dim0": 0.25, "center_dim1": 0.75},
                {"name": "bottom_left", "center_dim0": 0.75, "center_dim1": 0.25},
                {"name": "bottom_right", "center_dim0": 0.75, "center_dim1": 0.75}
            ]

    def add_custom_rectangular_regions(self, rectangles):
        """
        添加自定义矩形区域

        参数:
        rectangles: 矩形区域列表，格式：[[dim0_start, dim1_start, dim0_end, dim1_end], ...]
        """
        custom_regions = []
        for i, rect in enumerate(rectangles):
            dim0_start, dim1_start, dim0_end, dim1_end = rect

            # 计算中心点和区域大小
            center_dim0 = (dim0_start + dim0_end) / 2
            center_dim1 = (dim1_start + dim1_end) / 2

            custom_regions.append({
                "name": f"region_{i+1}",
                "center_dim0": center_dim0,
                "center_dim1": center_dim1,
                "dim0_start": dim0_start,
                "dim1_start": dim1_start,
                "dim0_end": dim0_end,
                "dim1_end": dim1_end
            })

        self.regions = custom_regions

    def detector(self, img1, img2, region=None, figpath=None):
        """
        便捷检测函数：支持文件名或numpy数组输入，自定义区域，返回[上,下,左,右]格式

        参数:
        img1: 参考图像（文件路径或numpy数组）
        img2: 目标图像（文件路径或numpy数组）
        region: 自定义检测区域列表，格式：[[dim0_start, dim1_start, dim0_end, dim1_end], ...]
        figpath: 可选，保存可视化图像的路径（包含区域框和箭头的img1）

        返回:
        list: [上, 下, 左, 右] 四个方向的移动程度（0-1之间的浮点数）
              - 上: abs(f_dim0) if f_dim0 < 0 else 0
              - 下: abs(f_dim0) if f_dim0 > 0 else 0
              - 左: abs(f_dim1) if f_dim1 < 0 else 0
              - 右: abs(f_dim1) if f_dim1 > 0 else 0

        示例:
        >>> detector = MotionDetector()
        >>> region = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        >>> result = detector.detector("img1.png", "img2.png", region, "output.png")
        >>> print(result)  # [0.0, 0.05, 0.0, 0.03] 表示向下0.05，向右0.03
        """
        # 1. 加载图像（支持文件名和numpy数组）
        image1 = self._load_image(img1)
        image2 = self._load_image(img2)

        if image1 is None or image2 is None:
            raise ValueError("Failed to load images. Check file paths or numpy arrays.")

        # 2. 设置自定义区域
        if region is not None:
            self.add_custom_rectangular_regions(region)

        # 3. 检测运动
        motion_result = self.detect_motion(image1, image2)

        # 4. 转换为 [上, 下, 左, 右] 格式
        f_dim0 = motion_result['f_dim0']
        f_dim1 = motion_result['f_dim1']

        up = abs(f_dim0) if f_dim0 < 0 else 0.0
        down = abs(f_dim0) if f_dim0 > 0 else 0.0
        left = abs(f_dim1) if f_dim1 < 0 else 0.0
        right = abs(f_dim1) if f_dim1 > 0 else 0.0

        result = [up, down, left, right]

        # 5. 可选：保存可视化结果
        if figpath:
            # 获取详细区域信息用于可视化
            detailed_regions_info = []
            for region_def in self.regions:
                region_result = self._detect_region_motion(image1, image2, region_def)
                if 'rect_coords' in region_result:
                    region_info = {
                        'region_name': region_result['region_name'],
                        'rect_coords': region_result['rect_coords'],
                        'confidence': region_result['confidence'],
                        'success': region_result['success']
                    }

                    # 添加airtest匹配的详细信息（如果有）
                    if 'ref_confidence' in region_result:
                        region_info['ref_confidence'] = region_result['ref_confidence']
                        region_info['search_confidence'] = region_result['search_confidence']

                    detailed_regions_info.append(region_info)

            # 绘制并保存可视化图像
            self.visualize_motion_on_image(
                image1,
                motion_result['f_dim1'],
                motion_result['f_dim0'],
                detailed_regions_info,
                save_path=figpath
            )

        return result

    def _load_image(self, img):
        """
        加载图像：支持文件路径（str）或numpy数组

        参数:
        img: 图像文件路径（str）或numpy数组

        返回:
        numpy.ndarray: 加载的图像，失败返回None
        """
        if isinstance(img, str):
            # 文件路径：使用cv2加载
            if not os.path.exists(img):
                print(f"Error: Image file not found: {img}")
                return None
            loaded_img = cv2.imread(img)
            if loaded_img is None:
                print(f"Error: Failed to load image: {img}")
            return loaded_img
        elif isinstance(img, np.ndarray):
            # numpy数组：直接返回
            return img
        else:
            print(f"Error: Unsupported image type: {type(img)}. Expected str or numpy.ndarray")
            return None

    def detect_motion(self, image1, image2):
        """
        检测两张图像之间的人物移动方向

        参数:
        image1: 参考图像
        image2: 目标图像

        返回:
        dict: {
            'f_dim1': float,       # dim1方向人物移动比例（背景移动的相反方向）
            'f_dim0': float,       # dim0方向人物移动比例（背景移动的相反方向）
            'direction': str,      # 人物移动方向：UP/DOWN/LEFT/RIGHT/NONE
            'confidence': float,   # 综合置信度
            'valid': bool          # 是否有效
        }

        注意：返回值已转换为人物移动方向（背景移动方向的相反方向）
        """
        results = []
        valid_results = []

        # 对每个区域进行检测
        for region_def in self.regions:
            result = self._detect_region_motion(image1, image2, region_def)
            results.append(result)

            if result['success']:
                valid_results.append(result)

        # 计算最终移动（加权平均）
        if len(valid_results) >= 2:
            valid_results.sort(key=lambda x: x['confidence'], reverse=True)
            top_results = valid_results[:3]

            total_confidence = sum(r['confidence'] for r in top_results)
            if total_confidence > 0:
                fx = sum(r['confidence'] * r['norm_dx'] for r in top_results) / total_confidence
                fy = sum(r['confidence'] * r['norm_dy'] for r in top_results) / total_confidence
                confidence = total_confidence / len(top_results)
                valid = True
            else:
                fx = fy = confidence = 0.0
                valid = False
        elif len(valid_results) == 1:
            fx = valid_results[0]['norm_dx']
            fy = valid_results[0]['norm_dy']
            confidence = valid_results[0]['confidence']
            valid = True
        else:
            fx = fy = confidence = 0.0
            valid = False

        # 将背景移动方向转换为人物移动方向（取反）
        hero_fx = -fx
        hero_fy = -fy

        direction = self._get_direction(hero_fx, hero_fy)

        return {
            'f_dim1': hero_fx,  # dim1方向人物移动（背景移动的相反方向）
            'f_dim0': hero_fy,  # dim0方向人物移动（背景移动的相反方向）
            'direction': direction,
            'confidence': confidence,
            'valid': valid
        }

    def _detect_region_motion(self, image1, image2, region_def):
        """
        检测单个区域的移动
        支持矩形区域和中心点区域两种模式
        """
        # 检查是否为矩形区域（有dim0_start等字段）
        if 'dim0_start' in region_def:
            # 矩形区域模式
            return self._detect_rectangular_region_motion(image1, image2, region_def)
        else:
            # 中心点区域模式（兼容原有逻辑）
            return self._detect_center_region_motion(image1, image2, region_def)

    def _detect_rectangular_region_motion(self, image1, image2, region_def):
        """
        检测矩形区域的移动
        """
        h, w = image1.shape[:2]

        # 转换为像素坐标
        dim0_start = int(region_def['dim0_start'] * h)
        dim1_start = int(region_def['dim1_start'] * w)
        dim0_end = int(region_def['dim0_end'] * h)
        dim1_end = int(region_def['dim1_end'] * w)

        # 提取矩形区域
        template = image1[dim0_start:dim0_end, dim1_start:dim1_end]

        # 在第二张图像中搜索，传递第一张图像作为参考
        search_result = self._search_template_rectangular(template, image2,
                                                        (dim0_start, dim1_start, dim0_end, dim1_end),
                                                        ref_image=image1)

        # 计算归一化偏移
        dx, dy = search_result['offset']
        norm_dx = dx / w
        norm_dy = dy / h

        result = {
            'region_name': region_def['name'],
            'offset': (dx, dy),
            'norm_dx': norm_dx,
            'norm_dy': norm_dy,
            'confidence': search_result['confidence'],
            'success': search_result['success'],
            'rect_coords': (dim0_start, dim1_start, dim0_end, dim1_end),  # 保存坐标用于可视化
            'confidence_normalized': search_result['confidence']
        }

        # 添加额外的匹配信息（如果有的话）
        if 'ref_pos' in search_result:
            result['ref_pos'] = search_result['ref_pos']
            result['search_pos'] = search_result['search_pos']
            result['ref_confidence'] = search_result['ref_confidence']
            result['search_confidence'] = search_result['search_confidence']

        return result

    def _detect_center_region_motion(self, image1, image2, region_def):
        """
        检测中心点区域的移动（原有逻辑）
        """
        # 提取区域
        template, template_info = self._extract_region(image1, region_def)

        # 在第二张图像中搜索
        search_result = self._search_template(template, image2, template_info)

        # 计算归一化偏移
        h, w = image2.shape[:2]
        dx, dy = search_result['offset']
        norm_dx = dx / w
        norm_dy = dy / h

        return {
            'region_name': region_def['name'],
            'offset': (dx, dy),
            'norm_dx': norm_dx,
            'norm_dy': norm_dy,
            'confidence': search_result['confidence'],
            'success': search_result['success']
        }

    def _extract_region(self, image, region_def):
        """
        从图像中提取指定区域
        """
        h, w = image.shape[:2]
        short_side = min(h, w)
        region_size = int(self.region_size_ratio * short_side)

        center_x = int(region_def["center_x"] * w)
        center_y = int(region_def["center_y"] * h)

        x1 = max(0, center_x - region_size // 2)
        y1 = max(0, center_y - region_size // 2)
        x2 = min(w, x1 + region_size)
        y2 = min(h, y1 + region_size)

        region_info = {
            'center': (center_x, center_y),
            'size': region_size,
            'bounds': (x1, y1, x2, y2)
        }

        return image[y1:y2, x1:x2], region_info

    def _search_template_rectangular(self, template, search_image, rect_coords, ref_image=None):
        """
        在搜索图像中寻找矩形模板区域
        使用airtest的Template匹配算法提高准确性
        """
        if self.use_airtest and ref_image is not None:
            return self._search_with_airtest_template(template, search_image, rect_coords, ref_image)
        else:
            return self._search_with_correlation(template, search_image, rect_coords)

    def _search_with_airtest_template(self, template, search_image, rect_coords, ref_image):
        """
        使用airtest的Template匹配算法
        """
        # 获取原始矩形坐标
        dim0_start, dim1_start, dim0_end, dim1_end = rect_coords
        center_dim0 = (dim0_start + dim0_end) // 2
        center_dim1 = (dim1_start + dim1_end) // 2

        try:
            # 创建临时文件保存template图像
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, template)
                template_path = tmp_file.name

            # 创建Template对象
            template_obj = Template(template_path, threshold=0.1)  # 使用低阈值获取更多结果

            # 在参考图像中搜索
            ref_match = template_obj.match_in(ref_image)
            ref_confidence = 0.0
            ref_pos = None

            if ref_match:
                # 从返回结果中提取置信度
                # match_in返回的是焦点位置，我们需要重新匹配获取完整结果
                try:
                    ref_result = template_obj._cv_match(ref_image)
                    if ref_result:
                        ref_confidence = ref_result.get('confidence', 0.0)
                        ref_pos = (ref_result['result'][0], ref_result['result'][1])
                except:
                    pass

            # 在搜索图像中搜索
            search_match = template_obj.match_in(search_image)
            search_confidence = 0.0
            search_pos = None

            if search_match:
                try:
                    search_result = template_obj._cv_match(search_image)
                    if search_result:
                        search_confidence = search_result.get('confidence', 0.0)
                        search_pos = (search_result['result'][0], search_result['result'][1])
                except:
                    pass

            # 清理临时文件
            os.unlink(template_path)

            # 计算偏移
            if ref_pos and search_pos:
                offset_dim1 = search_pos[0] - ref_pos[0]
                offset_dim0 = search_pos[1] - ref_pos[1]

                # 权重等于两次权重的乘积
                combined_confidence = ref_confidence * search_confidence

                return {
                    'success': combined_confidence > 0.3,  # 降低阈值
                    'confidence': max(0.0, combined_confidence),
                    'offset': (offset_dim1, offset_dim0),
                    'ref_pos': ref_pos,
                    'search_pos': search_pos,
                    'ref_confidence': ref_confidence,
                    'search_confidence': search_confidence
                }
            else:
                return {
                    'success': False,
                    'confidence': 0.0,
                    'offset': (0, 0)
                }

        except Exception as e:
            print(f"Airtest matching failed: {e}")
            # 回退到相关系数匹配
            return self._search_with_correlation(template, search_image, rect_coords)

    def _search_with_correlation(self, template, search_image, rect_coords):
        """
        使用相关系数匹配（原有方法）
        """
        # 预处理
        template_processed = self._preprocess(template)
        search_processed = self._preprocess(search_image)

        h_temp, w_temp = template_processed.shape
        h_search, w_search = search_processed.shape

        # 使用更大的搜索范围（因为矩形区域可能较大）
        base_search_range = min(self.search_range * 2, h_search - h_temp, w_search - w_temp)
        search_range = max(10, base_search_range)  # 最小10像素

        if search_range < 5:
            return {'success': False, 'confidence': 0.0, 'offset': (0, 0)}

        best_score = -1
        best_pos = None

        # 获取原始矩形坐标
        dim0_start, dim1_start, dim0_end, dim1_end = rect_coords
        center_dim0 = (dim0_start + dim0_end) // 2
        center_dim1 = (dim1_start + dim1_end) // 2

        # 在搜索范围内寻找最佳匹配位置
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                # 计算搜索窗口位置
                new_dim0_start = max(0, dim0_start + dy)
                new_dim1_start = max(0, dim1_start + dx)
                new_dim0_end = min(h_search, dim0_end + dy)
                new_dim1_end = min(w_search, dim1_end + dx)

                # 确保窗口大小正确
                if (new_dim0_end - new_dim0_start) != (dim0_end - dim0_start) or \
                   (new_dim1_end - new_dim1_start) != (dim1_end - dim1_start):
                    continue

                # 提取搜索窗口
                search_window = search_processed[new_dim0_start:new_dim0_end, new_dim1_start:new_dim1_end]

                # 确保大小匹配
                if search_window.shape != template_processed.shape:
                    continue

                # 计算相似度
                try:
                    correlation = np.corrcoef(template_processed.flatten(), search_window.flatten())[0, 1]
                    if not np.isnan(correlation):
                        score = (correlation + 1) / 2  # 归一化到[0,1]

                        if score > best_score:
                            best_score = score
                            best_pos = (new_dim1_start + (new_dim1_end - new_dim1_start) // 2,
                                       new_dim0_start + (new_dim0_end - new_dim0_start) // 2)
                except Exception:
                    continue

        if best_pos is not None and best_score > 0.5:
            # 计算偏移
            offset_dim1 = best_pos[0] - center_dim1
            offset_dim0 = best_pos[1] - center_dim0

            return {
                'success': True,
                'confidence': max(0.0, best_score),
                'offset': (offset_dim1, offset_dim0)
            }
        else:
            return {
                'success': False,
                'confidence': max(0.0, best_score),
                'offset': (0, 0)
            }

    def _search_template(self, template, search_image, template_info):
        """
        在搜索图像中寻找模板
        """
        # 预处理
        template_processed = self._preprocess(template)
        search_processed = self._preprocess(search_image)

        h_temp, w_temp = template_processed.shape
        h_search, w_search = search_processed.shape

        # 限制搜索范围
        search_range = min(self.search_range, w_search - w_temp, h_search - h_temp)
        if search_range < 5:
            return {'success': False, 'confidence': 0.0, 'offset': (0, 0)}

        best_score = -1
        best_pos = None
        center_x, center_y = template_info['center']

        # 搜索最佳匹配位置
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                # 计算搜索窗口位置
                x1 = max(0, center_x + dx - w_temp // 2)
                y1 = max(0, center_y + dy - h_temp // 2)
                x2 = min(w_search, x1 + w_temp)
                y2 = min(h_search, y1 + h_temp)

                # 确保窗口大小正确
                if x2 - x1 != w_temp or y2 - y1 != h_temp:
                    continue

                # 提取搜索窗口
                search_window = search_processed[y1:y2, x1:x2]

                # 计算相似度
                try:
                    correlation = np.corrcoef(template_processed.flatten(), search_window.flatten())[0, 1]
                    if not np.isnan(correlation):
                        score = (correlation + 1) / 2  # 归一化到[0,1]

                        if score > best_score:
                            best_score = score
                            best_pos = (x1 + w_temp // 2, y1 + h_temp // 2)
                except Exception:
                    continue

        if best_pos is not None and best_score > 0.5:
            # 计算偏移
            offset_x = best_pos[0] - center_x
            offset_y = best_pos[1] - center_y

            return {
                'success': True,
                'confidence': max(0.0, best_score),
                'offset': (offset_x, offset_y)
            }
        else:
            return {
                'success': False,
                'confidence': max(0.0, best_score),
                'offset': (0, 0)
            }

    def _preprocess(self, image):
        """
        图像预处理：灰度化+边缘增强
        """
        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 1.0)

        # 边缘增强
        edges = cv2.Canny(gray, 50, 150)
        enhanced = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)

        return enhanced

    def visualize_motion_on_image(self, image, f_dim1, f_dim0, regions_info=None, save_path=None):
        """
        在图像上绘制人物移动方向箭头（仅在第一对图像使用）

        参数:
        image: 原始图像（第一张参考图像）
        f_dim1: dim1方向人物移动比例（已经是取反后的值）
        f_dim0: dim0方向人物移动比例（已经是取反后的值）
        regions_info: 区域信息，用于绘制矩形框
        save_path: 保存路径（可选）

        返回:
        np.ndarray: 带箭头的图像
        """
        # 复制图像避免修改原图
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # 绘制自定义矩形区域（如果有提供区域信息）
        if regions_info:
            for i, region in enumerate(regions_info):
                if 'rect_coords' in region:
                    dim0_start, dim1_start, dim0_end, dim1_end = region['rect_coords']

                    # 根据置信度选择颜色：高置信度绿色，低置信度红色
                    confidence = region.get('confidence', 0.0)
                    success = region.get('success', False)

                    if success and confidence > 0.7:
                        color = (0, 255, 0)  # 绿色 - 高置信度
                        thickness = 3
                    elif success and confidence > 0.5:
                        color = (0, 255, 255)  # 黄色 - 中等置信度
                        thickness = 2
                    else:
                        color = (0, 0, 255)  # 红色 - 低置信度或失败
                        thickness = 2

                    # 绘制矩形框
                    cv2.rectangle(vis_image, (dim1_start, dim0_start), (dim1_end, dim0_end),
                                 color, thickness)

                    # 添加区域标签和置信度
                    if 'ref_confidence' in region and 'search_confidence' in region:
                        # 显示airtest匹配的详细置信度
                        label = f"{region['region_name']}: {region['ref_confidence']:.2f}*{region['search_confidence']:.2f}={confidence:.2f}"
                    else:
                        # 显示传统匹配的置信度
                        label = f"{region['region_name']}: {confidence:.2f}"
                    cv2.putText(vis_image, label, (dim1_start, dim0_start - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 在矩形中心添加序号
                    center_x = (dim1_start + dim1_end) // 2
                    center_y = (dim0_start + dim0_end) // 2
                    cv2.putText(vis_image, str(i + 1), (center_x - 10, center_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 绘制中心参考点（红色圆圈，纯英文）
        center_x = w // 2
        center_y = h // 2
        cv2.circle(vis_image, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.circle(vis_image, (center_x, center_y), 10, (0, 0, 255), 2)

        # 增大箭头长度便于观察（使用更大的比例）
        # f_dim0 和 f_dim1 已经是人物移动方向（背景移动的相反方向）
        max_arrow_length = min(w, h) // 4  # 增大到1/4图像尺寸
        arrow_length_x = int(f_dim1 * max_arrow_length * 5)  # 放大5倍便于观察
        arrow_length_y = int(f_dim0 * max_arrow_length * 5)

        # 限制箭头长度，但允许更大的显示范围
        arrow_length_x = max(-max_arrow_length, min(max_arrow_length, arrow_length_x))
        arrow_length_y = max(-max_arrow_length, min(max_arrow_length, arrow_length_y))

        # 绘制水平箭头（绿色）- 表示人物在dim1方向的移动
        if abs(arrow_length_x) > 3:  # 降低最小长度限制
            start_point = (center_x, center_y)
            end_point = (center_x + arrow_length_x, center_y)
            cv2.arrowedLine(vis_image, start_point, end_point, (0, 255, 0), 4, tipLength=0.3)

        # 绘制垂直箭头（蓝色）- 表示人物在dim0方向的移动
        if abs(arrow_length_y) > 3:  # 降低最小长度限制
            start_point = (center_x, center_y)
            end_point = (center_x, center_y + arrow_length_y)
            cv2.arrowedLine(vis_image, start_point, end_point, (255, 0, 0), 4, tipLength=0.3)

        # 添加文字说明（纯英文，显示人物移动方向）
        text_y = 40
        cv2.putText(vis_image, f"Hero dim0={f_dim0:+.4f}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)  # dim0 蓝色

        text_y += 35
        cv2.putText(vis_image, f"Hero dim1={f_dim1:+.4f}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  # dim1 绿色

        text_y += 35
        hero_direction = self._get_direction(f_dim1, f_dim0)
        cv2.putText(vis_image, f"Hero Move: {hero_direction}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # 添加背景移动方向（与人物移动相反）
        text_y += 40
        bg_motion_dim0 = "DOWN" if f_dim0 < 0 else "UP" if f_dim0 > 0 else "NONE"
        bg_motion_dim1 = "RIGHT" if f_dim1 < 0 else "LEFT" if f_dim1 > 0 else "NONE"
        bg_direction = f"Background: [{bg_motion_dim0},{bg_motion_dim1}]"
        cv2.putText(vis_image, bg_direction, (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        # 简洁图例（纯英文，避免编码问题）
        text_y = h - 30
        cv2.putText(vis_image, "Arrow: Hero Movement (dim0| dim1->)", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # 保存可视化结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"  移动方向示意图已保存: {save_path}")

        return vis_image

    def _get_direction(self, f_dim1, f_dim0):
        """
        根据人物移动向量确定方向

        重要逻辑说明：
        1. 坐标系：dim0=垂直方向(从上到下)，dim1=水平方向(从左到右)
        2. 传入参数 f_dim0, f_dim1 已经是人物移动方向（背景移动的相反方向）
        3. 背景移动计算：background_motion = 图2位置 - 图1位置
        4. 人物移动：hero_motion = -background_motion（在detect_motion中已转换）
        5. 本函数直接根据人物移动方向返回指令：
           - f_dim0>0 → 英雄向下移动 → DOWN
           - f_dim0<0 → 英雄向上移动 → UP
           - f_dim1>0 → 英雄向右移动 → RIGHT
           - f_dim1<0 → 英雄向左移动 → LEFT

        改进：使用相对比例判断主导方向，避免dim1只比dim0大一点点就判定为水平移动
        """
        min_threshold = 0.001

        if abs(f_dim1) < min_threshold and abs(f_dim0) < min_threshold:
            return "NONE"

        # 计算相对比例，考虑主导性
        abs_dim1 = abs(f_dim1)
        abs_dim0 = abs(f_dim0)

        # 如果两个方向的移动都很小，返回"NONE"
        if abs_dim1 < min_threshold * 2 and abs_dim0 < min_threshold * 2:
            return "NONE"

        # 使用比例判断主导方向，要求一个方向明显大于另一个方向
        # 比例阈值：需要至少1.5倍的差异才判定为主导方向
        ratio_threshold = 1.5

        if abs_dim1 > abs_dim0 * ratio_threshold:
            # dim1方向明显为主导 - 直接根据人物移动方向返回
            # f_dim1>0 → 英雄向右移动 → RIGHT
            # f_dim1<0 → 英雄向左移动 → LEFT
            return "RIGHT" if f_dim1 > 0 else "LEFT"
        elif abs_dim0 > abs_dim1 * ratio_threshold:
            # dim0方向明显为主导 - 直接根据人物移动方向返回
            # f_dim0>0 → 英雄向下移动 → DOWN
            # f_dim0<0 → 英雄向上移动 → UP
            return "DOWN" if f_dim0 > 0 else "UP"
        else:
            # 两个方向移动幅度相近，根据较大者判断人物移动方向
            if abs_dim1 > abs_dim0:
                return "RIGHT" if f_dim1 > 0 else "LEFT"
            else:
                return "DOWN" if f_dim0 > 0 else "UP"


def natural_sort_key(filename):
    """
    自然排序键函数
    """
    import re
    parts = re.split(r'(_|\d+)', filename)
    result = []
    for part in parts:
        if part == '':
            continue
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part.lower())
    return result

def main():
    """
    主函数：测试移动检测
    """
    import argparse

    parser = argparse.ArgumentParser(description='Image Motion Direction Detector')
    parser.add_argument('--image_folder', type=str, default='pic_data',
                       help='图像文件夹路径')
    parser.add_argument('--region_size', type=float, default=0.15,
                       help='检测区域大小比例（默认: 0.15）')
    parser.add_argument('--search_range', type=int, default=40,
                       help='搜索范围（默认: 40）')
    parser.add_argument('--use_airtest', action='store_true', default=True,
                       help='使用airtest的Template匹配算法（默认: True）')
    parser.add_argument('--no_airtest', action='store_true',
                       help='禁用airtest匹配，使用传统相关系数匹配')

    args = parser.parse_args()

    # 确定是否使用airtest
    use_airtest = args.use_airtest and not args.no_airtest

    print("=== Image Motion Direction Detection ===")
    print(f"Image folder: {args.image_folder}")
    print(f"Region size: {args.region_size}")
    print(f"Search range: {args.search_range}")
    print(f"Use airtest matching: {use_airtest}")
    print()

    # 检查文件夹
    if not os.path.exists(args.image_folder):
        print(f"Error: Folder does not exist: {args.image_folder}")
        return

    # 获取图像文件
    image_files = [f for f in os.listdir(args.image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(image_files) < 2:
        print(f"Error: Need at least 2 images, found: {len(image_files)}")
        return

    # 自然排序
    image_files.sort(key=natural_sort_key)
    print(f"Found {len(image_files)} images")

    # 定义自定义矩形区域（你提供的坐标）
    # 格式：[dim0_start, dim1_start, dim0_end, dim1_end]
    # 王者荣耀使用这个region最佳
    custom_rectangles = [
        [0.128, 0.367, 0.302, 0.543],  # 区域1
        [0.231, 0.568, 0.413, 0.744],  # 区域2
        [0.347, 0.325, 0.647, 0.419],  # 区域3
        [0.647, 0.285, 0.827, 0.475]   # 区域4
    ]

    # 创建检测器并使用自定义区域
    detector = MotionDetector(
        region_size=args.region_size,
        search_range=args.search_range,
        use_airtest=use_airtest
    )

    # 添加自定义矩形区域
    detector.add_custom_rectangular_regions(custom_rectangles)

    # 处理图像对
    results = []
    for i in range(len(image_files) - 1):
        img1_path = os.path.join(args.image_folder, image_files[i])
        img2_path = os.path.join(args.image_folder, image_files[i + 1])

        try:
            image1 = cv2.imread(img1_path)
            image2 = cv2.imread(img2_path)

            if image1 is None or image2 is None:
                print(f"Warning: Failed to load images {img1_path} or {img2_path}")
                continue

            # 只在第一对图像时输出详细信息
            if i == 0:
                m, n = image1.shape[:2]  # m=dim0, n=dim1
                print(f"Reference image shape: [{m},{n},3] = (dim0, dim1, RGB)")
                print(f"坐标系: dim0=垂直方向(从上到下), dim1=水平方向(从左到右)")
                print(f"返回值: f_dim0, f_dim1 = 人物移动方向（背景移动的相反方向）")
                print(f"游戏逻辑: 相机跟随英雄，背景移动与英雄移动相反")
                print()

            result = detector.detect_motion(image1, image2)

            # 输出结果（第一对详细输出，其他简洁格式化）
            if i == 0:
                # 第一对：详细输出
                h2, w2 = image2.shape[:2]
                pixel_dx = result['f_dim1'] * w2
                pixel_dy = result['f_dim0'] * h2

                # 计算背景移动方向（与人物移动相反）
                bg_motion_dim0 = "DOWN" if result['f_dim0'] < 0 else "UP" if result['f_dim0'] > 0 else "NONE"
                bg_motion_dim1 = "RIGHT" if result['f_dim1'] < 0 else "LEFT" if result['f_dim1'] > 0 else "NONE"

                output = f"{image_files[i]} -> {image_files[i+1]}: "
                output += f"f_dim0={result['f_dim0']:.4f}({pixel_dy:+.1f}px), f_dim1={result['f_dim1']:.4f}({pixel_dx:+.1f}px)\n"
                output += f"    人物移动: {result['direction']}, "
                output += f"背景移动: [{bg_motion_dim0}, {bg_motion_dim1}]"
                if result['valid']:
                    output += f", confidence={result['confidence']:.3f}"
                else:
                    output += ", invalid"
                print(output)
            else:
                # 其他对：简洁格式化输出（保持与第一对相同的dim0,dim1顺序）
                print(f"{image_files[i]} -> {image_files[i+1]}: f_dim0={result['f_dim0']:.4f}, f_dim1={result['f_dim1']:.4f}, {result['direction']}")

            # 只在第一对图像时生成可视化，并传递区域信息用于绘制矩形
            if i == 0:
                # 重新运行检测以获取详细区域信息
                detailed_regions_info = []
                for j, region_def in enumerate(detector.regions):
                    region_result = detector._detect_region_motion(image1, image2, region_def)
                    if 'rect_coords' in region_result:
                        # 合并区域定义和检测结果
                        region_info = {
                            'region_name': region_result['region_name'],
                            'rect_coords': region_result['rect_coords'],
                            'confidence': region_result['confidence'],
                            'success': region_result['success']
                        }

                        # 添加airtest匹配的详细信息
                        if 'ref_confidence' in region_result:
                            region_info['ref_confidence'] = region_result['ref_confidence']
                            region_info['search_confidence'] = region_result['search_confidence']
                            region_info['ref_pos'] = region_result['ref_pos']
                            region_info['search_pos'] = region_result['search_pos']

                            # 获取区域坐标
                            dim0_start, dim1_start, dim0_end, dim1_end = region_result['rect_coords']
                            center_dim0 = (dim0_start + dim0_end) // 2
                            center_dim1 = (dim1_start + dim1_end) // 2

                            print(f"  Region {j+1} ({region_result['region_name']}):")
                            print(f"    区域坐标: [{dim0_start}:{dim0_end}, {dim1_start}:{dim1_end}], 中心: ({center_dim0}, {center_dim1})")
                            print(f"    图1位置: {region_result['ref_pos']}")
                            print(f"    图2位置: {region_result['search_pos']}")
                            print(f"    位置偏移: ({region_result['search_pos'][0] - region_result['ref_pos'][0]}, "
                                  f"{region_result['search_pos'][1] - region_result['ref_pos'][1]})")
                            print(f"    ref_conf={region_result['ref_confidence']:.3f}, "
                                  f"search_conf={region_result['search_confidence']:.3f}, "
                                  f"combined={region_result['confidence']:.3f}")
                        else:
                            # 传统匹配的位置信息
                            dim0_start, dim1_start, dim0_end, dim1_end = region_result['rect_coords']
                            center_dim0 = (dim0_start + dim0_end) // 2
                            center_dim1 = (dim1_start + dim1_end) // 2
                            dx, dy = region_result['offset']

                            print(f"  Region {j+1} ({region_result['region_name']}):")
                            print(f"    区域坐标: [{dim0_start}:{dim0_end}, {dim1_start}:{dim1_end}], 中心: ({center_dim0}, {center_dim1})")
                            print(f"    图1中心: ({center_dim1}, {center_dim0})")
                            print(f"    图2中心: ({center_dim1 + dx}, {center_dim0 + dy})")
                            print(f"    位置偏移: ({dx}, {dy})")
                            print(f"    confidence={region_result['confidence']:.3f}")

                        detailed_regions_info.append(region_info)

                vis_image = detector.visualize_motion_on_image(image1, result['f_dim1'], result['f_dim0'], detailed_regions_info)
                # 固定文件名
                vis_filename = "motion_vis.png"
                cv2.imwrite(vis_filename, vis_image)  # 保存在当前目录
                print(f"  Motion visualization saved: {vis_filename}")

            results.append(result)

        except Exception as e:
            print(f"Error: Processing {image_files[i]} -> {image_files[i+1]} failed: {e}")

    # 统计
    if results:
        valid_results = [r for r in results if r['valid']]
        print(f"\n=== Statistics ===")
        print(f"Total: {len(results)}, Valid: {len(valid_results)}")
        if valid_results:
            directions = {}
            for r in valid_results:
                direction = r['direction']
                directions[direction] = directions.get(direction, 0) + 1
            print("Direction statistics:")
            for direction, count in sorted(directions.items()):
                print(f"  {direction}: {count}")

if __name__ == "__main__":
    main()