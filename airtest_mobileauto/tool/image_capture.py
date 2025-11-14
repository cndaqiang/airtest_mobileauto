from airtest_mobileauto.control import  Settings, deviceOB
import cv2
import os
from datetime import datetime
import threading
import time

class ImageCapture:
    """图像捕获工具类 - 用于截图和图像处理"""

    def __init__(self, airtest_config="config.yaml", save_folder="pic_data"):
        self.airtest_config = airtest_config
        self.save_folder = save_folder
        # 创建保存文件夹如果不存在
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def airtest_init(self):
        """初始化airtest连接"""
        Settings.Config(self.airtest_config)
        # device
        self.mynode = Settings.mynode
        self.totalnode = Settings.totalnode
        self.totalnode_bak = self.totalnode
        self.LINK = Settings.LINK_dict[Settings.mynode]
        self.mobile_device = deviceOB(mynode=self.mynode, totalnode=self.totalnode, LINK=self.LINK)
        self.airtest = True

    def screenshot_airtest(self, filename=None):
        """
        使用 Airtest 截取屏幕并返回图像数据。

        参数:
        filename (str, optional): 保存图片的文件名，如果为None则不保存

        返回:
        np.ndarray: 截图的图像数据。
        """
        try:
            arr = self.mobile_device.device.snapshot()
            print("---> screenshot_airtest: Screenshot successful")

            # 如果指定了文件名，保存图片
            if filename:
                filepath = os.path.join(self.save_folder, filename)
                cv2.imwrite(filepath, arr)
                print(f"---> Image saved to: {filepath}")
                # 返回文件路径而不是数组，便于判断保存是否成功
                return filepath

            # 调试保存
            cv2.imwrite("screenshot_airtest.png", arr)
            # 如果没有指定文件名，返回截图数组
            return arr
        except Exception as e:
            print(e)
            return None

    def save_image_with_auto_name(self, image, prefix="screenshot"):
        """
        自动保存图片到指定文件夹，文件名包含时间戳

        参数:
        image (np.ndarray): 要保存的图像数据
        prefix (str): 文件名前缀

        返回:
        str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.save_folder, filename)
        cv2.imwrite(filepath, image)
        print(f"---> Image auto-saved to: {filepath}")
        return filepath

    def load_image(self, filename):
        """
        从指定文件夹加载图片

        参数:
        filename (str): 图片文件名

        返回:
        np.ndarray: 加载的图像数据，如果失败返回None
        """
        try:
            filepath = os.path.join(self.save_folder, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"---> Failed to load image: {filepath}")
                return None
            print(f"---> Successfully loaded image: {filepath}")
            return image
        except Exception as e:
            print(f"---> Failed to load image: {e}")
            return None

    def start_enter_capture(self, prefix="debug"):
        """
        开始监听回车键截图，在新线程中运行

        参数:
        prefix (str): 保存文件名前缀
        """
        def capture_loop():
            print("---> Starting enter key capture listener, press Enter to capture, ESC to exit...")
            while True:
                try:
                    # 简单的输入监听
                    user_input = input("Press Enter to capture, type 'q' to quit: ")
                    if user_input.lower() == 'q':
                        print("---> Capture listener stopped")
                        break

                    # 截图
                    screenshot = self.screenshot_airtest()
                    if screenshot is not None:
                        filepath = self.save_image_with_auto_name(screenshot, prefix)
                        print(f"---> Screenshot saved: {filepath}")
                    else:
                        print("---> Screenshot failed")

                except KeyboardInterrupt:
                    print("---> Capture listener stopped")
                    break
                except Exception as e:
                    print(f"---> Error during capture: {e}")

        # 在新线程中运行监听
        capture_thread = threading.Thread(target=capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        return capture_thread

    def quick_save_screenshot(self, prefix="debug"):
        """
        快速截图并保存，用于调试时手动调用

        参数:
        prefix (str): 保存文件名前缀

        返回:
        str: 保存的文件路径
        """
        screenshot = self.screenshot_airtest()
        if screenshot is not None:
            return self.save_image_with_auto_name(screenshot, prefix)
        return None


def main():
    """
    纯截图工具主函数
    功能：
    1. 初始化连接
    2. 批量截图
    3. 简单保存到指定文件夹

    注意：这是一个纯截图工具，不包含任何检测功能
    如需检测功能，请使用 test_image_detector.py
    """
    print("=== Pure Screenshot Tool ===")
    print("This is a pure capture tool without detection functionality")
    print("For detection features, please use test_image_detector.py")

    try:
        # 创建截图工具实例
        capture_tool = ImageCapture(save_folder="pic_data")

        # 步骤1: 初始化连接
        print("\n--- Step 1: Initializing Connection ---")
        capture_tool.airtest_init()
        print("[OK] Connection successful")

        # 测试基础截图
        print("\n--- Testing Basic Screenshot ---")
        screenshot = capture_tool.screenshot_airtest()
        if screenshot is not None:
            print(f"[OK] Screenshot successful, size: {screenshot.shape}")

            # 保存测试截图
            test_path = capture_tool.save_image_with_auto_name(screenshot, "connection_test")
            print(f"[OK] Test screenshot saved: {test_path}")
        else:
            print("[ERROR] Screenshot failed, please check connection")
            return

        # 步骤2: 批量截图模式
        print("\n--- Step 2: Batch Capture Mode ---")
        print("Instructions:")
        print("- Press Enter: Capture and save")
        print("- Type number + Enter: Wait for N seconds then capture")
        print("- Type 'q' + Enter: Quit")
        print("- Type 'l' + Enter: List capture commands")
        print("\nNote: Images will be saved to pic_data/ folder")

        capture_count = 0

        while True:
            user_input = input(f"\n[{capture_count}] Enter command: ").strip()

            if user_input.lower() == 'q':
                print("Exiting batch capture mode")
                break

            if user_input.lower() == 'l':
                print("Available commands:")
                print("  Enter: Capture immediately")
                print("  N + Enter: Wait N seconds then capture")
                print("  q: Quit")
                print("  l: List commands")
                continue

            # 检查是否输入了数字（等待时间）
            if user_input.isdigit():
                wait_time = int(user_input)
                print(f"Waiting {wait_time} seconds before capture...")
                time.sleep(wait_time)

            # 执行截图
            print("Capturing...")
            filepath = capture_tool.quick_save_screenshot(f"capture_{capture_count}")

            if filepath:
                print(f"[OK] Screenshot successful: {filepath}")
                capture_count += 1

                # 每5张截图给出提示
                if capture_count % 5 == 0:
                    print(f"\n[STAT] Captured {capture_count} screenshots")
                    print("[TIP] Suggest capturing different states:")
                    print("   - Normal game scenes")
                    print("   - Different interfaces")
                    print("   - Various game states")
                    print("   - After detection, use test_image_detector.py for analysis")
            else:
                print("[ERROR] Screenshot failed")

        # 总结
        print(f"\n=== Batch Capture Complete ===")
        print(f"[OK] Total screenshots: {capture_count}")
        print(f"[OK] Images saved to: pic_data/")
        print("[OK] Ready for analysis with test_image_detector.py")

    except Exception as e:
        print(f"[ERROR] Batch capture failed: {e}")
        print("Please check connection and configuration")

if __name__ == "__main__":
    main()

# 向后兼容，保持原有导入
auto = ImageCapture