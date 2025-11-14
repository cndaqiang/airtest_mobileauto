"""
图像状态检测器测试脚本
仿照test_hero_status.py的结构

功能:
1. 设定区域，循环设置阈值，对pic_xxx文件夹数据进行判断
2. 快速判断，使用默认阈值
3. 保存第一张图并标记检测区域
4. 格式化输出和错误文件统计
"""

from autowzry.tool.image_detector import ImageDetector
import sys
import os
import cv2
from datetime import datetime
import glob
import shutil

def get_input(prompt, default_value, value_type=str):
    """获取用户输入，支持默认值"""
    result = input(prompt) or str(default_value)
    try:
        return value_type(result)
    except:
        return default_value

def select_regions():
    """让用户选择检测区域"""
    print("\n=== 选择检测区域 ===")
    print("1. 仅技能区域 (50%, 90%)")
    print("2. 仅普攻区域 (91%, 86%)")
    print("3. 两个区域都使用")
    print("4. 自定义区域")

    choice = input("\n请选择区域 (1/2/3/4, 回车使用1): ").strip()

    if choice == "1" or choice == "":
        return [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]
    elif choice == "2":
        return [{"name": "attack_button", "center_dim0": 0.86, "center_dim1": 0.91}]
    elif choice == "3":
        return [
            {"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50},
            {"name": "attack_button", "center_dim0": 0.86, "center_dim1": 0.91}
        ]
    elif choice == "4":
        regions = []
        while True:
            name = input("输入区域名称 (或按回车结束): ").strip()
            if not name:
                break
            center_dim0 = get_input("中心点dim0坐标 (垂直方向0-1): ", 0.9, float)
            center_dim1 = get_input("中心点dim1坐标 (水平方向0-1): ", 0.5, float)
            regions.append({"name": name, "center_dim0": center_dim0, "center_dim1": center_dim1})
        return regions if regions else [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]
    else:
        return [{"name": "skill_button", "center_dim0": 0.90, "center_dim1": 0.50}]

def threshold_tuning_mode(detector, light_folder, dark_folder):
    """阈值调优模式"""
    print("\n=== 阈值调优模式 ===")

    # 设置阈值扫描范围
    print("\n设置阈值扫描范围（按回车使用默认值）")
    threshold_start = get_input("起始阈值 (默认60): ", 60, int)
    threshold_end = get_input("结束阈值 (默认100): ", 100, int)
    threshold_step = get_input("步长 (默认5): ", 5, int)

    print(f"\n阈值扫描范围: {threshold_start}-{threshold_end}, 步长={threshold_step}")

    # 扫描不同阈值
    best_threshold = 80
    best_accuracy = 0
    results_summary = []

    for threshold in range(threshold_start, threshold_end + 1, threshold_step):
        detector.set_threshold(threshold)

        # 测试存活图片
        alive_results = detector.evaluate_folder(light_folder, expected_status=True, verbose=False)
        alive_correct = sum(1 for r in alive_results if r['correct'])
        alive_total = len(alive_results)

        # 测试死亡图片
        dead_results = detector.evaluate_folder(dark_folder, expected_status=False, verbose=False)
        dead_correct = sum(1 for r in dead_results if r['correct'])
        dead_total = len(dead_results)

        # 计算准确率
        total_correct = alive_correct + dead_correct
        total_images = alive_total + dead_total
        accuracy = (total_correct / total_images * 100) if total_images > 0 else 0

        results_summary.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'alive_correct': alive_correct,
            'alive_total': alive_total,
            'dead_correct': dead_correct,
            'dead_total': dead_total
        })

        print(f"阈值 {threshold}: 准确率 {accuracy:.1f}% (存活 {alive_correct}/{alive_total}, 死亡 {dead_correct}/{dead_total})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # 显示最佳结果
    print(f"\n=== 最佳阈值 ===")
    print(f"最佳阈值: {best_threshold}")
    print(f"最佳准确率: {best_accuracy:.1f}%")

    # 使用前5名阈值重新测试并显示详细信息
    top_results = sorted(results_summary, key=lambda x: x['accuracy'], reverse=True)[:5]
    print(f"\n前5个最佳阈值:")
    for i, result in enumerate(top_results):
        print(f"{i+1}. 阈值={result['threshold']}, 准确率={result['accuracy']:.1f}%")

    # 使用最佳阈值进行详细测试
    if best_accuracy > 0:
        print(f"\n使用最佳阈值 {best_threshold} 进行详细测试:")
        detector.set_threshold(best_threshold)
        return best_threshold

    return 80  # 默认值

def copy_by_score(detector, source_folder, high_threshold=80, low_threshold=60):
    """
    根据得分将图片复制到高分和低分文件夹

    参数:
    detector: 检测器实例
    source_folder: 源图片文件夹
    high_threshold: 高分阈值（大于等于此值复制到高分文件夹）
    low_threshold: 低分阈值（小于此值复制到低分文件夹）
    """
    print(f"\n=== Copy by Score Mode ===")
    print(f"Source folder: {source_folder}")
    print(f"High score threshold: {high_threshold}")
    print(f"Low score threshold: {low_threshold}")

    if not os.path.exists(source_folder):
        print(f"[ERROR] Source folder not found: {source_folder}")
        return

    # 创建目标文件夹
    high_folder = os.path.join(source_folder, "high_score")
    low_folder = os.path.join(source_folder, "low_score")

    for folder in [high_folder, low_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[OK] Created folder: {folder}")

    # 获取所有图片文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(source_folder, ext)))

    if not image_files:
        print(f"[ERROR] No image files found in {source_folder}")
        return

    print(f"[OK] Found {len(image_files)} images to process")

    high_count = 0
    low_count = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)

        try:
            # 计算得分
            score = detector.calculate_score(img_path)

            # 根据得分分类
            if score >= high_threshold:
                # 高分态 - 复制到高分文件夹
                dest_path = os.path.join(high_folder, filename)
                high_count += 1
                status = "HIGH"
            elif score < low_threshold:
                # 低分态 - 复制到低分文件夹
                dest_path = os.path.join(low_folder, filename)
                low_count += 1
                status = "LOW"
            else:
                # 中间分数 - 不复制
                print(f"  - {filename}: Score={score:.1f} (MIDDLE - skipped)")
                continue

            # 复制文件
            import shutil
            shutil.copy2(img_path, dest_path)
            print(f"  + {filename}: Score={score:.1f} -> {status}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    print(f"\n=== Copy Complete ===")
    print(f"[OK] Total processed: {len(image_files)}")
    print(f"[OK] High score images: {high_count} (saved to {high_folder})")
    print(f"[OK] Low score images: {low_count} (saved to {low_folder})")
    print(f"[OK] Ready for testing with these folders")

def quick_test_mode(detector, light_folder, dark_folder):
    """快速测试模式"""
    print(f"\n=== 快速测试模式 ===")
    print(f"使用阈值: {detector.get_threshold()}")

    # 测试存活图片
    print(f"\n测试存活图片 ({light_folder}):")
    alive_results = detector.evaluate_folder(light_folder, expected_status=True, verbose=True)

    # 测试死亡图片
    print(f"\n测试死亡图片 ({dark_folder}):")
    dead_results = detector.evaluate_folder(dark_folder, expected_status=False, verbose=True)

    # 统计结果
    alive_correct = sum(1 for r in alive_results if r['correct'])
    alive_total = len(alive_results)
    dead_correct = sum(1 for r in dead_results if r['correct'])
    dead_total = len(dead_results)

    total_correct = alive_correct + dead_correct
    total_images = alive_total + dead_total
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0

    print(f"\n=== 测试结果统计 ===")
    print(f"存活图片: {alive_correct}/{alive_total} = {alive_correct/alive_total*100:.1f}%")
    print(f"死亡图片: {dead_correct}/{dead_total} = {dead_correct/dead_total*100:.1f}%")
    print(f"总体准确率: {total_correct}/{total_images} = {overall_accuracy:.1f}%")

    # 输出判断错误的文件
    print(f"\n=== 判断错误的文件 ===")
    error_files = []

    # 状态A误判为状态B
    for result in alive_results:
        if not result['correct']:
            error_files.append({
                'filename': result['filename'],
                'score': result['score'],
                'expected': '状态A',
                'predicted': '状态B',
                'folder': light_folder
            })

    # 状态B误判为状态A
    for result in dead_results:
        if not result['correct']:
            error_files.append({
                'filename': result['filename'],
                'score': result['score'],
                'expected': '状态B',
                'predicted': '状态A',
                'folder': dark_folder
            })

    if error_files:
        print(f"共 {len(error_files)} 个文件判断错误:")
        for error in error_files:
            print(f"  {error['folder']}/{error['filename']}: 得分={error['score']:.1f}, 期望={error['expected']}, 预测={error['predicted']}")
    else:
        print("所有文件判断正确！")

    return alive_results + dead_results

def visualize_mode(detector, image_path, save_path):
    """可视化模式"""
    print(f"\n=== 可视化模式 ===")

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return

    # 可视化并保存
    result_image = detector.visualize_regions(image_path, save_path)

    # 显示区域信息
    image = cv2.imread(image_path)
    positions = detector.get_region_positions(image.shape)

    print(f"检测区域信息:")
    for pos in positions:
        print(f"  {pos['name']}: 位置=({pos['dim0_start']},{pos['dim1_start']})-({pos['dim0_end']},{pos['dim1_end']}), 大小={pos['size']}x{pos['size']}")

    print(f"可视化结果已保存: {save_path}")

def main():
    print("=" * 60)
    print("图像状态检测器测试")
    print("=" * 60)

    # 选择检测区域
    regions = select_regions()

    # 创建检测器
    detector = ImageDetector(regions=regions, threshold=80)

    print(f"\n创建检测器，使用 {len(regions)} 个检测区域:")
    for region in regions:
        print(f"  {region['name']}: ({region['center_dim0']}, {region['center_dim1']})")

    # 选择测试模式
    print(f"\n=== 选择测试模式 ===")
    print("1. 阈值调优（寻找最佳阈值）")
    print("2. 快速测试（使用默认阈值）")
    print("3. 可视化检测区域")
    print("4. 按得分复制（自动分类图片）")

    mode = input("\n请输入选择 (1/2/3/4, 回车使用2): ").strip() or "2"

    if mode == "1":
        # 阈值调优模式
        best_threshold = threshold_tuning_mode(detector, "pic_data/high_score", "pic_data/low_score")
        detector.set_threshold(best_threshold)

        # 使用最佳阈值进行详细测试
        print(f"\n" + "=" * 60)
        print(f"使用最佳阈值 {best_threshold} 进行详细测试:")
        quick_test_mode(detector, "pic_data/high_score", "pic_data/low_score")

    elif mode == "2":
        # 快速测试模式
        results = quick_test_mode(detector, "pic_data/high_score", "pic_data/low_score")

    elif mode == "3":
        # 可视化模式
        # 选择要可视化的图片
        print(f"\n选择要可视化的图片:")
        print("1. 使用第一张存活图片")
        print("2. 使用第一张死亡图片")
        print("3. 输入自定义路径")

        choice = input("\n请输入选择 (1/2/3, 回车使用1): ").strip() or "1"

        if choice == "1":
            # 找到第一张存活图片
            if os.path.exists("pic_data/high_score"):
                files = [f for f in os.listdir("pic_data/high_score") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    image_path = os.path.join("pic_data/high_score", files[0])
                else:
                    print("未找到高分态图片，使用pic_data文件夹")
                    # 如果没有分类文件夹，使用默认文件夹
                    if os.path.exists("pic_data"):
                        files = [f for f in os.listdir("pic_data") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if files:
                            image_path = os.path.join("pic_data", files[0])
                        else:
                            print("未找到任何图片")
                            return
                    else:
                        print("pic_data 文件夹不存在")
                        return
            else:
                print("pic_data/high_score 文件夹不存在")
                # 使用默认文件夹
                if os.path.exists("pic_data"):
                    files = [f for f in os.listdir("pic_data") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if files:
                        image_path = os.path.join("pic_data", files[0])
                    else:
                        print("未找到任何图片")
                        return
                else:
                    print("pic_data 文件夹不存在")
                    return
        elif choice == "2":
            # 找到第一张死亡图片
            if os.path.exists("pic_data/low_score"):
                files = [f for f in os.listdir("pic_data/low_score") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    image_path = os.path.join("pic_data/low_score", files[0])
                else:
                    print("未找到低分态图片，使用pic_data文件夹")
                    # 如果没有分类文件夹，使用默认文件夹
                    if os.path.exists("pic_data"):
                        files = [f for f in os.listdir("pic_data") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if files:
                            image_path = os.path.join("pic_data", files[0])
                        else:
                            print("未找到任何图片")
                            return
                    else:
                        print("pic_data 文件夹不存在")
                        return
            else:
                print("pic_data/low_score 文件夹不存在")
                # 使用默认文件夹
                if os.path.exists("pic_data"):
                    files = [f for f in os.listdir("pic_data") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if files:
                        image_path = os.path.join("pic_data", files[0])
                    else:
                        print("未找到任何图片")
                        return
                else:
                    print("pic_data 文件夹不存在")
                    return
        else:
            image_path = input("请输入图片路径: ").strip()

        # 生成保存路径
        save_path = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # 可视化
        visualize_mode(detector, image_path, save_path)

    elif mode == "4":
        # 按得分复制模式
        print(f"\n=== 按得分复制模式 ===")
        print("此模式将根据图片得分自动分类到high_score和low_score文件夹")

        # 选择源文件夹
        print(f"\n选择源图片文件夹:")
        print("1. 使用pic_data文件夹（推荐）")
        print("2. 输入自定义路径")

        folder_choice = input("\n请输入选择 (1/2, 回车使用1): ").strip() or "1"

        if folder_choice == "1":
            source_folder = "pic_data"
        else:
            source_folder = input("请输入源文件夹路径: ").strip()

        if not os.path.exists(source_folder):
            print(f"[ERROR] 源文件夹不存在: {source_folder}")
            return

        # 设置阈值
        high_threshold = get_input("高分阈值 (默认80): ", 80, int)
        low_threshold = get_input("低分阈值 (默认60): ", 60, int)

        print(f"\n开始按得分复制...")
        print(f"高分阈值: {high_threshold}, 低分阈值: {low_threshold}")

        # 执行复制
        copy_by_score(detector, source_folder, high_threshold, low_threshold)

        # 询问是否立即测试新分类的图片
        test_new = input("\n是否立即测试新分类的图片? (y/n, 默认=y): ").lower() == 'y' or input("") == ""

        if test_new:
            high_folder = os.path.join(source_folder, "high_score")
            low_folder = os.path.join(source_folder, "low_score")

            if os.path.exists(high_folder) and os.path.exists(low_folder):
                print(f"\n=== 测试新分类的图片 ===")
                results = quick_test_mode(detector, high_folder, low_folder)
            else:
                print("[ERROR] 找不到分类后的文件夹")

    else:
        print("无效的选择")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)