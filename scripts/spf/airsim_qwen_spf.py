"""
最简 AirSim + Qwen-VL 实现
依赖: airsim, openai, opencv-python, numpy

使用方法:
  pip install airsim openai opencv-python numpy
  1. 启动 AirSim 仿真器（Unreal Engine + AirSim插件）
  2. python airsim_qwen.py
"""

import cv2
import numpy as np
import base64
import json
import math
import time
import airsim
from openai import OpenAI

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
QWEN_API_KEY  = "your-api-key"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-vl-max"

CAMERA_NAME      = "0"     # AirSim 摄像头名称
BASE_VELOCITY    = 2.0     # m/s 基础飞行速度
BASE_YAW_RATE    = 30.0    # °/s 基础偏航速率
MIN_DURATION     = 2.0     # s 最短动作持续时间


# ─────────────────────────────────────────────
# 1. Qwen-VL 客户端
# ─────────────────────────────────────────────
class QwenVLClient:
    def __init__(self):
        self.client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    def ask(self, image_bgr: np.ndarray, instruction: str) -> dict | None:
        _, buf = cv2.imencode(".jpg", image_bgr)
        b64 = base64.b64encode(buf).decode()

        prompt = f"""你是无人机导航专家，分析无人机摄像头画面。

任务：{instruction}

找到画面中与任务描述最匹配的目标，在目标中心放一个点，并估计深度。

只返回如下 JSON，不要其他文字：
[{{"point": [y, x], "depth": 深度值, "label": "目标描述"}}]

坐标系：
- x: 0-1000（500=中心，>500=右，<500=左）
- y: 0-1000（0=顶部，1000=底部）
- depth: 1-10（1=很近/大，10=很远/小）"""

        try:
            resp = self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                temperature=0.4,
                max_tokens=256,
            )
            text = resp.choices[0].message.content.strip()
            for tag in ("```json", "```"):
                if tag in text:
                    text = text.split(tag)[1].split("```")[0].strip()
                    break
            data = json.loads(text)
            return data[0] if isinstance(data, list) else data

        except Exception as e:
            print(f"[Qwen-VL] 错误: {e}")
            return None


# ─────────────────────────────────────────────
# 2. 坐标投影
# ─────────────────────────────────────────────
def point_to_3d(px: int, py: int, depth: float,
                img_w: int, img_h: int, fov=108.0):
    """像素坐标 + 深度 → (dx, dy, dz)"""
    cx, cy = img_w / 2, img_h / 2
    half_fov = math.radians(fov / 2)
    x_norm = (px - cx) / cx
    y_norm = (cy - py) / cy  # 上为正

    dx = depth * x_norm * math.tan(half_fov)
    dz = depth * y_norm * math.tan(half_fov)
    dy = depth
    return dx, dy, dz


def depth_to_scale(vlm_depth: float) -> float:
    """VLM深度(1-10) → 飞行缩放因子（0表示停止）"""
    if vlm_depth <= 2:
        return 0.0   # 太近，停止
    elif vlm_depth <= 5:
        return vlm_depth / 10.0 * 2
    else:
        return 1.0 + (vlm_depth - 5) / 5.0


# ─────────────────────────────────────────────
# 3. AirSim 控制器
# ─────────────────────────────────────────────
class AirSimController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("已连接 AirSim")

    def takeoff(self):
        print("起飞...")
        self.client.takeoffAsync().join()
        print("起飞完成")

    def capture(self, camera=CAMERA_NAME) -> np.ndarray | None:
        """截取仿真摄像头画面"""
        try:
            responses = self.client.simGetImages(
                [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
            )
            if not responses or responses[0].width == 0:
                return None
            r = responses[0]
            arr = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
            return arr.reshape(r.height, r.width, 3)
        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def get_image_size(self, camera=CAMERA_NAME):
        """获取摄像头分辨率"""
        responses = self.client.simGetImages(
            [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
        )
        if responses and responses[0].width > 0:
            return responses[0].width, responses[0].height
        return 640, 480

    def move(self, dx: float, dy: float, dz: float, depth_scale: float):
        """执行移动：先旋转，再前进"""
        if depth_scale == 0:
            print("  [动作] 目标太近，停止移动")
            return

        velocity = BASE_VELOCITY * depth_scale
        yaw_rate = BASE_YAW_RATE * depth_scale

        dist_xy = math.sqrt(dx**2 + dy**2)

        # 1. 旋转对准目标
        if dist_xy > 0.01:
            target_angle = math.degrees(math.atan2(dx, dy))
            if target_angle > 180:
                target_angle -= 360
            elif target_angle < -180:
                target_angle += 360

            if abs(target_angle) > 10:
                duration = abs(target_angle) / yaw_rate
                rate = yaw_rate if target_angle > 0 else -yaw_rate
                print(f"  [动作] 旋转 {target_angle:.1f}° ({duration:.1f}s)")
                self.client.rotateByYawRateAsync(rate, duration).join()
                self.client.hoverAsync().join()

        # 2. 前进（沿机头方向）
        if dist_xy > 0.01:
            duration = max(dist_xy / velocity, MIN_DURATION)
            vz = (-dz / dist_xy) * velocity if dist_xy > 0.01 else 0
            print(f"  [动作] 前进 v={velocity:.1f}m/s 持续 {duration:.1f}s vz={vz:.2f}")
            self.client.moveByVelocityBodyFrameAsync(
                velocity, 0, vz, duration,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=0)
            ).join()
            self.client.hoverAsync().join()

    def stop(self):
        self.client.hoverAsync().join()
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("已降落，控制权已释放")


# ─────────────────────────────────────────────
# 4. 主程序
# ─────────────────────────────────────────────
def main():
    instruction = input("请输入飞行指令（如：穿越吊车结构）：").strip()
    if not instruction:
        instruction = "向前飞行，避开障碍物"

    drone = AirSimController()
    vlm = QwenVLClient()

    img_w, img_h = drone.get_image_size()
    print(f"摄像头分辨率: {img_w}×{img_h}")
    if img_w < 640:
        print("⚠️ 分辨率较低，建议在 AirSim settings.json 中调整摄像头设置")

    try:
        print("3秒后起飞...")
        time.sleep(3)
        drone.takeoff()

        print(f"\n开始执行：{instruction}")
        print("Ctrl+C 退出\n")

        while True:
            frame = drone.capture()
            if frame is None:
                print("截图失败，等待...")
                time.sleep(1)
                continue

            print("─" * 40)
            print("正在分析画面...")
            result = vlm.ask(frame, instruction)

            if result is None:
                print("VLM 未返回有效结果，跳过")
                time.sleep(1)
                continue

            y_norm, x_norm = result["point"]
            vlm_depth = result.get("depth", 5)
            label = result.get("label", "目标")

            px = int(x_norm / 1000.0 * img_w)
            py = int(y_norm / 1000.0 * img_h)

            depth_scale = depth_to_scale(vlm_depth)
            dx, dy, dz = point_to_3d(px, py, max(depth_scale, 0.1), img_w, img_h)

            print(f"目标: {label}")
            print(f"位置: ({x_norm}, {y_norm}) 深度: {vlm_depth}/10 缩放: {depth_scale:.2f}")
            print(f"3D向量: ({dx:.2f}, {dy:.2f}, {dz:.2f})")

            drone.move(dx, dy, dz, depth_scale)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        drone.stop()


if __name__ == "__main__":
    main()
