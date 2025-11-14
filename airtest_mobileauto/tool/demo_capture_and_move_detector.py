#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时截图并识别方向工具
结合 ImageCapture 和 MotionDetector 功能
每5张保存一次连续2张截图，第一张标注检测区域和方向箭头
"""

import cv2
import os
import time
from datetime import datetime
from image_capture import ImageCapture
from image_motion_detector import MotionDetector


class CaptureAndDetect:
    """
    实时截图并识别方向的工具类
    每N张保存一次连续两张图像,第一张标注方向和区域
    """

    def __init__(self, airtest_config="config.yaml", save_folder="pic_data",
                 save_interval=5, custom_rectangles=None,
                 region_size=0.15, search_range=40, use_airtest=True):
        """
        初始化工具

        参数:
        airtest_config: airtest配置文件路径
        save_folder: 图像保存文件夹
        save_interval: 每隔多少张保存一次(默认5)
        custom_rectangles: 自定义检测矩形区域列表
        region_size: 检测区域大小比例
        search_range: 搜索范围(像素)
        use_airtest: 是否使用airtest的Template匹配
        """
        # 初始化截图工具
        self.capture_tool = ImageCapture(airtest_config, save_folder)
        self.save_folder = save_folder
        self.save_interval = save_interval

        # 创建保存文件夹
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # 初始化运动检测器
        self.detector = MotionDetector(
            region_size=region_size,
            search_range=search_range,
            use_airtest=use_airtest
        )

        # 设置自定义矩形区域(如果提供)
        if custom_rectangles:
            self.detector.add_custom_rectangular_regions(custom_rectangles)
        else:
            # 使用默认区域(推荐根据你的应用场景自定义)
            default_rectangles = [
                [0.128, 0.367, 0.302, 0.543],  # 区域1
                [0.231, 0.568, 0.413, 0.744],  # 区域2
                [0.347, 0.325, 0.647, 0.419],  # 区域3
                [0.647, 0.285, 0.827, 0.475]   # 区域4
            ]
            self.detector.add_custom_rectangular_regions(default_rectangles)

        # 用于存储上一张截图
        self.previous_image = None
        self.capture_count = 0

        # 用于存储检测结果历史
        self.direction_history = []

    def airtest_init(self):
        """初始化airtest连接"""
        self.capture_tool.airtest_init()
        print("[OK] Airtest connection initialized")

    def capture_and_detect_once(self):
        """
        执行一次截图和方向检测

        返回:
        dict: {
            'image': 当前截图,
            'direction': 检测到的方向,
            'f_dim0': dim0方向移动比例,
            'f_dim1': dim1方向移动比例,
            'confidence': 置信度,
            'valid': 是否有效,
            'regions_info': 区域检测详细信息
        }
        """
        # 截图
        current_image = self.capture_tool.screenshot_airtest()
        if current_image is None:
            print("[ERROR] Screenshot failed")
            return None

        self.capture_count += 1

        # 如果是第一张图,保存为参考图像
        if self.previous_image is None:
            self.previous_image = current_image
            return {
                'image': current_image,
                'direction': 'NONE',
                'f_dim0': 0.0,
                'f_dim1': 0.0,
                'confidence': 0.0,
                'valid': False,
                'regions_info': None
            }

        # 检测移动方向
        motion_result = self.detector.detect_motion(self.previous_image, current_image)

        # 获取详细区域信息
        detailed_regions_info = []
        for region_def in self.detector.regions:
            region_result = self.detector._detect_region_motion(
                self.previous_image, current_image, region_def
            )
            if 'rect_coords' in region_result:
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

                detailed_regions_info.append(region_info)

        # 准备返回结果
        result = {
            'image': current_image,
            'previous_image': self.previous_image,
            'direction': motion_result['direction'],
            'f_dim0': motion_result['f_dim0'],
            'f_dim1': motion_result['f_dim1'],
            'confidence': motion_result['confidence'],
            'valid': motion_result['valid'],
            'regions_info': detailed_regions_info
        }

        # 更新参考图像
        self.previous_image = current_image

        # 记录方向历史
        if motion_result['valid']:
            self.direction_history.append(motion_result['direction'])

        return result

    def should_save_pair(self):
        """
        判断是否应该保存图像对

        返回:
        bool: True表示应该保存
        """
        return self.capture_count % self.save_interval == 0

    def save_image_pair(self, result, prefix="pair"):
        """
        保存连续两张图像，第一张标注检测区域和方向箭头

        参数:
        result: capture_and_detect_once返回的结果字典
        prefix: 文件名前缀

        返回:
        tuple: (标注图像路径, 第二张图像路径)
        """
        if result is None or result['previous_image'] is None:
            print("[WARNING] No valid image pair to save")
            return None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_id = self.capture_count // self.save_interval

        # 在第一张图像上绘制检测区域和方向箭头
        annotated_image = self.detector.visualize_motion_on_image(
            result['previous_image'],
            result['f_dim1'],
            result['f_dim0'],
            result['regions_info']
        )

        # 保存第一张(标注后的图像)
        annotated_path = os.path.join(
            self.save_folder,
            f"{prefix}_{pair_id:03d}_1_annotated_{timestamp}.png"
        )
        cv2.imwrite(annotated_path, annotated_image)

        # 保存第二张(原始图像)
        original_path = os.path.join(
            self.save_folder,
            f"{prefix}_{pair_id:03d}_2_original_{timestamp}.png"
        )
        cv2.imwrite(original_path, result['image'])

        print(f"[SAVE] Image pair {pair_id} saved:")
        print(f"  Annotated: {annotated_path}")
        print(f"  Original:  {original_path}")
        print(f"  Direction: {result['direction']}, Confidence: {result['confidence']:.3f}")

        return annotated_path, original_path

    def run_continuous(self, max_captures=None, interval_seconds=0.5, auto_save=True):
        """
        连续���行截图和检测

        参数:
        max_captures: 最大截图数量，None表示无限制
        interval_seconds: 截图间隔(秒)
        auto_save: 是否自动保存图像对(默认True)

        返回:
        list: 检测结果列表
        """
        print("=== Starting Continuous Capture and Detection ===")
        print(f"Save interval: Every {self.save_interval} captures")
        print(f"Capture interval: {interval_seconds} seconds")
        print(f"Max captures: {max_captures if max_captures else 'Unlimited'}")
        print(f"Auto save: {auto_save}")
        print("\nPress Ctrl+C to stop\n")

        results = []
        pair_count = 0

        try:
            while True:
                # 检查是否达到最大截图数量
                if max_captures and self.capture_count >= max_captures:
                    print(f"\n[STOP] Reached max captures: {max_captures}")
                    break

                # 执行一次截图和检测
                result = self.capture_and_detect_once()

                if result:
                    results.append(result)

                    # 输出检测结果
                    if result['valid']:
                        print(f"[{self.capture_count}] Direction: {result['direction']}, "
                              f"f_dim0={result['f_dim0']:+.4f}, f_dim1={result['f_dim1']:+.4f}, "
                              f"confidence={result['confidence']:.3f}")
                    else:
                        print(f"[{self.capture_count}] No valid motion detected")

                    # 检查是否应该保存图像对
                    if auto_save and self.should_save_pair():
                        self.save_image_pair(result, prefix="auto_pair")
                        pair_count += 1

                # 等待下一次截图
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n[STOP] Interrupted by user")

        # 输出统计信息
        print("\n=== Summary ===")
        print(f"Total captures: {self.capture_count}")
        print(f"Saved pairs: {pair_count}")
        print(f"Valid detections: {len([r for r in results if r['valid']])}")

        if self.direction_history:
            # 统计方向分布
            direction_count = {}
            for direction in self.direction_history:
                direction_count[direction] = direction_count.get(direction, 0) + 1

            print("\nDirection statistics:")
            for direction, count in sorted(direction_count.items()):
                percentage = (count / len(self.direction_history)) * 100
                print(f"  {direction}: {count} ({percentage:.1f}%)")

        return results

    def run_interactive(self):
        """
        交互式运行模式
        按Enter截图并检测，自动保存图像对
        """
        print("=== Interactive Capture and Detection ===")
        print("Commands:")
        print("  Enter: Capture and detect")
        print("  's': Save current pair manually")
        print("  'q': Quit")
        print(f"\nAuto-save: Every {self.save_interval} captures\n")

        try:
            while True:
                user_input = input(f"[{self.capture_count}] Command: ").strip().lower()

                if user_input == 'q':
                    print("Exiting...")
                    break

                # 执行截图和检测
                result = self.capture_and_detect_once()

                if result:
                    # 显示结果
                    if result['valid']:
                        print(f"  Direction: {result['direction']}, "
                              f"f_dim0={result['f_dim0']:+.4f}, f_dim1={result['f_dim1']:+.4f}, "
                              f"confidence={result['confidence']:.3f}")
                    else:
                        print("  No valid motion detected")

                    # 自动保存
                    if self.should_save_pair():
                        self.save_image_pair(result, prefix="auto_pair")
                        print(f"  [AUTO-SAVE] Pair saved at capture #{self.capture_count}")

                    # 手动保存
                    if user_input == 's' and result['previous_image'] is not None:
                        self.save_image_pair(result, prefix="manual_pair")
                        print("  [MANUAL-SAVE] Pair saved")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        # 输出统计
        print("\n=== Summary ===")
        print(f"Total captures: {self.capture_count}")
        if self.direction_history:
            print(f"Valid detections: {len(self.direction_history)}")
            direction_count = {}
            for direction in self.direction_history:
                direction_count[direction] = direction_count.get(direction, 0) + 1
            print("Direction distribution:")
            for direction, count in sorted(direction_count.items()):
                print(f"  {direction}: {count}")


def main():
    """
    主函数：实时截图并检测方向
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Capture screenshots and detect motion direction in real-time'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Airtest config file path (default: config.yaml)')
    parser.add_argument('--save_folder', type=str, default='pic_data',
                        help='Folder to save images (default: pic_data)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save pair every N captures (default: 5)')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Interval between captures in seconds (default: 0.5)')
    parser.add_argument('--max_captures', type=int, default=None,
                        help='Maximum number of captures (default: unlimited)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--region_size', type=float, default=0.15,
                        help='Detection region size ratio (default: 0.15)')
    parser.add_argument('--search_range', type=int, default=40,
                        help='Search range in pixels (default: 40)')
    parser.add_argument('--no_airtest', action='store_true',
                        help='Disable airtest Template matching')

    args = parser.parse_args()

    # 定义检测区域(可以根据实际应用调整)
    custom_rectangles = [
        [0.128, 0.367, 0.302, 0.543],  # 区域1
        [0.231, 0.568, 0.413, 0.744],  # 区域2
        [0.347, 0.325, 0.647, 0.419],  # 区域3
        [0.647, 0.285, 0.827, 0.475]   # 区域4
    ]

    try:
        # 创建工具实例
        tool = CaptureAndDetect(
            airtest_config=args.config,
            save_folder=args.save_folder,
            save_interval=args.save_interval,
            custom_rectangles=custom_rectangles,
            region_size=args.region_size,
            search_range=args.search_range,
            use_airtest=not args.no_airtest
        )

        # 初始化连接
        print("Initializing Airtest connection...")
        tool.airtest_init()
        print("[OK] Connection successful\n")

        # 运行模式
        if args.interactive:
            tool.run_interactive()
        else:
            tool.run_continuous(
                max_captures=args.max_captures,
                interval_seconds=args.interval,
                auto_save=True
            )

    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
