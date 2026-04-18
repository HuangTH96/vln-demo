"""
最简 Tello + Qwen-VL 实现
依赖: djitellopy, openai, opencv-python, numpy
Qwen-VL 通过 OpenAI 兼容接口调用（DashScope 或本地部署均可）

使用方法:
  pip install djitellopy openai opencv-python numpy
  python tello_qwen.py
"""

import cv2
import numpy as np
import base64
import json
import math
import time
import threading
from openai import OpenAI
from djitellopy import Tello

# ─────────────────────────────────────────────
# 配置（按需修改）
# ─────────────────────────────────────────────
QWEN_API_KEY  = "your-api-key"               # DashScope API Key
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-vl-max"               # 或 qwen-vl-plus

TELLO_SPEED   = 60    # RC速度 0-100
ROTATE_TIME   = 3400  # ms：以speed=100时旋转360°所需时间，需按实际speed调整
MOVE_TIME     = 500   # ms：移动1个单位距离所需时间


# ─────────────────────────────────────────────
# 1. Qwen-VL 客户端
# ─────────────────────────────────────────────
class QwenVLClient:
    def __init__(self):
        self.client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    def ask(self, image_bgr: np.ndarray, instruction: str) -> dict | None:
        """
        发送图像+指令，返回解析后的 JSON dict，失败返回 None
        """
        _, buf = cv2.imencode(".jpg", image_bgr)
        b64 = base64.b64encode(buf).decode()

        # TODO： 从prompt可以看到，这是一个visual servoing，一步一动，不具备路径规划能力
        # TODO：输出多个点，进行轨迹规划？
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

            # 去除可能的 markdown 代码块
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
# 2. 坐标投影：2D点 → 3D向量
# ─────────────────────────────────────────────
def point_to_3d(px: int, py: int, depth: float,
                img_w=960, img_h=720, fov=108.0):
    """
    将像素坐标 + 深度转为 (dx, dy, dz) 3D向量
    dx=左右, dy=前后(深度), dz=上下

    (px-img_w/2) / (img_w/2) → x_norm (-1~1) = dx / (d * tan(fov/2))
    """
    ref_y = img_h * 0.35  # 参考水平线（略高于中心）
    x_norm = (px - img_w / 2) / (img_w / 2)
    y_norm = (ref_y - py) / (img_h / 2)

    depth_factor = 1.0 + y_norm * 0.5
    d = depth * depth_factor

    half_fov = math.radians(fov / 2)
    dx = d * x_norm * math.tan(half_fov)
    dz = d * y_norm * math.tan(half_fov)
    dy = d
    return dx, dy, dz

def depth_to_scale(vlm_depth: float):
    """
    将VLM深度(1-10)转为实际飞行缩放比，同时判断是否太近
    返回 (adjusted_depth, yaw_only)
    """
    if vlm_depth <= 2:
        return 0.5, True   # 太近：只转向不前进
    base = (vlm_depth / 10.0) ** 2.0 * 8
    return base, False

# ─────────────────────────────────────────────
# 3. 动作执行：3D向量 → Tello RC命令
# ─────────────────────────────────────────────
def execute_action(tello: Tello, dx: float, dy: float, dz: float,
                   yaw_only: bool = False):
    """
    将3D向量转为 Tello RC 控制命令并执行
    """
    rotate_time_adj = ROTATE_TIME * (100 / TELLO_SPEED)  # 按速度调整时间
    move_time_adj   = MOVE_TIME   * (100 / TELLO_SPEED)

    if yaw_only:
        # 只执行偏航
        target_angle = math.degrees(math.atan2(dx, dy)) % 360
        if target_angle > 180:
            dur = int((360 - target_angle) * rotate_time_adj / 360)
            print(f"  [动作] 左转 {360-target_angle:.1f}° ({dur}ms)")
            tello.send_rc_control(-TELLO_SPEED, 0, 0, 0)  # yaw 用 send_rc
            # 注意：Tello yaw 是第4个参数
            tello.send_rc_control(0, 0, 0, -TELLO_SPEED)
            time.sleep(dur / 1000.0)    # 运动时长
        else:
            dur = int(target_angle * rotate_time_adj / 360)
            print(f"  [动作] 右转 {target_angle:.1f}° ({dur}ms)")
            tello.send_rc_control(0, 0, 0, TELLO_SPEED)
            time.sleep(dur / 1000.0)
        tello.send_rc_control(0, 0, 0, 0)
        return

    # 1. 偏航对准目标
    target_angle = math.degrees(math.atan2(dx, dy)) % 360
    if abs(dx) > 0.05 or abs(dy) > 0.05:
        if target_angle > 180:
            dur = int((360 - target_angle) * rotate_time_adj / 360)
            print(f"  [动作] 左转 {360-target_angle:.1f}° ({dur}ms)")
            tello.send_rc_control(0, 0, 0, -TELLO_SPEED)
        else:
            dur = int(target_angle * rotate_time_adj / 360)
            print(f"  [动作] 右转 {target_angle:.1f}° ({dur}ms)")
            tello.send_rc_control(0, 0, 0, TELLO_SPEED)
        time.sleep(dur / 1000.0)
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)

    # 2. 前进
    dist_xy = math.sqrt(dx**2 + dy**2)
    if dist_xy > 0.05:
        dur = int(dist_xy * move_time_adj)
        print(f"  [动作] 前进 {dur}ms")
        tello.send_rc_control(0, TELLO_SPEED, 0, 0)
        time.sleep(dur / 1000.0)
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)

    # 3. 升降
    if abs(dz) > 0.05:
        dur = int(abs(dz) * move_time_adj)
        if dz > 0:
            print(f"  [动作] 上升 {dur}ms")
            tello.send_rc_control(0, 0, TELLO_SPEED, 0)
        else:
            print(f"  [动作] 下降 {dur}ms")
            tello.send_rc_control(0, 0, -TELLO_SPEED, 0)
        time.sleep(dur / 1000.0)
        tello.send_rc_control(0, 0, 0, 0)


# ─────────────────────────────────────────────
# 4. 摄像头帧获取
# ─────────────────────────────────────────────
class FrameGrabber:
    """后台线程持续抓取最新帧"""
    def __init__(self, tello: Tello):
        self.tello = tello
        self.frame = None
        self._lock = threading.Lock()
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            try:
                fr = self.tello.get_frame_read().frame
                if fr is not None:
                    with self._lock:
                        self.frame = fr.copy()
            except Exception:
                pass
            time.sleep(0.05)

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
# 5. 主程序
# ─────────────────────────────────────────────
def main():
    instruction = input("请输入飞行指令（如：飞向红色门）：").strip()
    if not instruction:
        instruction = "向前飞行避开障碍物"

    # 连接 Tello
    tello = Tello()
    tello.connect()
    print(f"电量: {tello.get_battery()}%")
    tello.streamon()

    grabber = FrameGrabber(tello)
    vlm = QwenVLClient()

    # 等待摄像头就绪
    print("等待摄像头初始化...")
    for _ in range(10):
        time.sleep(1)
        frame = grabber.get()
        if frame is not None and np.std(frame) > 10:
            print("摄像头就绪！")
            break

    try:
        print("5秒后起飞...")
        time.sleep(5)
        tello.takeoff()
        time.sleep(2)

        print(f"\n开始执行：{instruction}")
        print("Ctrl+C 退出\n")

        while True:
            frame = grabber.get()
            if frame is None:
                print("未获取到帧，等待...")
                time.sleep(1)
                continue

            print("─" * 40)
            print("正在分析画面...")
            result = vlm.ask(frame, instruction)

            if result is None:
                print("VLM 未返回有效结果，跳过")
                time.sleep(1)
                continue

            # 解析结果
            y_norm, x_norm = result["point"]
            vlm_depth = result.get("depth", 5)
            label = result.get("label", "目标")

            px = int(x_norm / 1000.0 * 960)
            py = int(y_norm / 1000.0 * 720)

            adj_depth, yaw_only = depth_to_scale(vlm_depth)
            dx, dy, dz = point_to_3d(px, py, adj_depth)

            print(f"目标: {label}")
            print(f"位置: ({x_norm}, {y_norm}) 深度: {vlm_depth}/10")
            print(f"3D向量: ({dx:.2f}, {dy:.2f}, {dz:.2f})")
            if yaw_only:
                print("⚠️  目标太近，仅执行转向")

            execute_action(tello, dx, dy, dz, yaw_only=yaw_only)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        grabber.stop()
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        print("已降落，程序结束")


if __name__ == "__main__":
    main()
