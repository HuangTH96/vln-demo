import airsim

import cv2
import numpy as np
import math
import time
import pprint
import base64
import json
import os
from openai import OpenAI
from typing import Tuple

DEFAULT_SPEED      = 3.0    # 默认飞行速度 m/s
DEFAULT_ALTITUDE   = -5.0   # 默认飞行高度（NED坐标，负值=向上）
HOVER_DURATION     = 3.0    # 到达航点后悬停时间 s
IMAGE_CAMERA_ID    = "0"    # AirSim 相机 ID

# == High level planner: Qwen_vl_max
SYSTEM_PROMPT = """
你是一个无人机飞行控制助手。
用户会给你一段自然语言指令和当前无人机的状态信息，以及当前视角的图像。
你需要根据指令，生成一系列三维航点（waypoints）供无人机执行。

坐标系说明：
- X轴：沿着X轴方向为前进方向
- Y轴：沿着Y轴正方向为向右移动
- Z轴：沿着Z轴正方向为向上移动（负值=向上，如 z=-5 表示高度5米）

输出格式要求（严格JSON，不要有任何多余文字）：
{
  "waypoints": [
    {"x": 5.0, "y": 0.0, "z": -5.0, "speed": 3.0, "description": "向前飞5米"}
  ],
  "action_after": "hover",
  "summary": "向前飞5米，1个航点直达"
}

action_after 可选值：
- "hover"  : 完成后悬停
- "land"   : 完成后降落
- "return" : 完成后返回起点

航点规划原则：
- 每个航点之间的欧式距离不要超过1米
- 需要转向的路径：在转折点处增加航点
- 曲线路径：根据曲率自行判断需要多少航点来平滑过渡
- 要保证路径的准确性和平滑性
- 输出的航点序列中，不考虑当前位置

注意：
- 直接返回 JSON 格式的导航计划，不要包含任何 Markdown 代码块标记（如 ```json），不要包含任何解释性文字。
- 每个航点之间的欧式距离不要超过1米
- 航点坐标是相对于当前位置的累计偏移量（不是每段的增量）
- 输出的航点序列中，不考虑当前位置
- 速度单位为 m/s，建议范围 [0.0-0.5]
- 高度保持默认值除非任务需要
- 如果指令不清晰，返回一个原地转一圈的指令
- 每个航点都需要有清晰的 description 说明当前段的动作
"""

# == utils: retrive image 
def get_scene_image(client: airsim.MultirotorClient) -> Tuple[bytes, str]:
    """
    从 AirSim 获取当前场景图像
    返回: (numpy图像, base64字符串)
    """
    # simGetImage 直接返回压缩的 PNG 字节，更简单可靠
    png_image = client.simGetImage(IMAGE_CAMERA_ID, airsim.ImageType.Scene)

    # PNG 字节直接 base64 编码
    img_base64 = base64.b64encode(png_image).decode('utf-8')

    return png_image, img_base64


def get_depth_image(client: airsim.MultirotorClient) -> np.ndarray:
    """获取深度图像用于障碍物检测"""
    responses = client.simGetImages([
        airsim.ImageRequest(IMAGE_CAMERA_ID, airsim.ImageType.DepthVis, False, True)
    ])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_bgr = cv2.imdecode(img1d, cv2.IMREAD_UNCHANGED)
    return img_bgr

# == utils: retrive drone states
def get_drone_state(client: airsim.MultirotorClient) -> dict:
    """获取无人机当前状态"""
    state = client.getMultirotorState()
    pos   = state.kinematics_estimated.position
    vel   = state.kinematics_estimated.linear_velocity
    pitch, roll, yaw = airsim.to_eularian_angles(
        state.kinematics_estimated.orientation
    )
    return {
        "position": {"x": round(pos.x_val, 2),
                     "y": round(pos.y_val, 2),
                     "z": round(pos.z_val, 2)},
        "velocity": {"vx": round(vel.x_val, 2),
                     "vy": round(vel.y_val, 2),
                     "vz": round(vel.z_val, 2)},
        "attitude": {"pitch": round(math.degrees(pitch), 2),
                     "roll":  round(math.degrees(roll),  2),
                     "yaw":   round(math.degrees(yaw),   2)},
    }

def calling_qwen(
    instruction: str,
    drone_state: dict,
    img_base64: str,
    qwen_client,
    QWEN_MODEL="qwen-vl-max"
) -> dict:
    state_str = json.dumps(drone_state, ensure_ascii=False, indent=2)

    # 构造消息内容
    user_content = []

    # 有图像时附带图像
    if img_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })
    else:
        print("[警告] 无图像，仅使用文字状态推理")

    user_content.append({
        "type": "text",
        "text": (
            f"当前无人机状态：\n{state_str}\n\n"
            f"飞行指令：{instruction}\n\n"
            f"请根据图像中的环境信息和无人机状态，生成航点列表。"
        )
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    response = qwen_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.1,
    )

    return response

def parse_waypoints_from_vlm(
    instruction: str,
    drone_state: dict,
    img_base64: str,
    qwen_client
) -> dict:
    """
    调用 Qwen-VL API，根据指令、无人机状态和图像生成航点
    """
    print(f"Calling qwen...\n")
    response = calling_qwen(instruction, drone_state, img_base64, qwen_client)
    raw = response.choices[0].message.content.strip()

    start = raw.find('{')
    end   = raw.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f"VLM 输出中没有找到 JSON: {raw}")

    plan = json.loads(raw[start:end])
    return plan

def wait_until_reached(
    client: airsim.MultirotorClient,
    target_x: float,
    target_y: float,
    target_z: float,
    tolerance: float = 0.5,
    timeout: float = 60.0
) -> bool:
    """主动轮询，直到无人机真正到达目标位置"""
    time.sleep(1.0)  # 等待指令生效，让飞行器开始运动

    start = time.time()
    last_dist = None
    stuck_count = 0

    while time.time() - start < timeout:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        dist = math.sqrt(
            (pos.x_val - target_x) ** 2 +
            (pos.y_val - target_y) ** 2 +
            (pos.z_val - target_z) ** 2
        )
        speed = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)
        print(f"  [轮询] 距目标 {dist:.2f} m | 当前速度 {speed:.2f} m/s", end="\r")

        if dist < tolerance:
            print(f"\n  [到达] 误差 {dist:.2f} m")
            return True

        # 检测是否卡住（距离长时间没变化且速度接近0）
        if last_dist is not None and abs(last_dist - dist) < 0.05 and speed < 0.1:
            stuck_count += 1
            if stuck_count >= 10:  # 连续2秒没动
                print(f"\n  [卡住] 飞行器停止运动，距目标 {dist:.2f} m，重新发送指令")
                return False
        else:
            stuck_count = 0

        last_dist = dist
        time.sleep(0.2)

    print(f"\n  [超时] {timeout}s 内未到达")
    return False

def execute_waypoints(client: airsim.MultirotorClient, plan: dict) -> None:
    """
    执行 VLM 生成的航点列表
    """
    waypoints    = plan.get("waypoints", [])
    action_after = plan.get("action_after", "hover")
    summary      = plan.get("summary", "")

    if not waypoints:
        print("[执行] 没有航点，悬停")
        client.hoverAsync().join()
        return

    print(f"\n[执行] 任务摘要: {summary}")
    print(f"[执行] 共 {len(waypoints)} 个航点，完成后执行: {action_after}\n")

    state    = client.getMultirotorState()
    base_pos = state.kinematics_estimated.position
    base_x   = base_pos.x_val
    base_y   = base_pos.y_val

    for idx, wp in enumerate(waypoints):
        target_x = base_x + wp["x"]
        target_y = base_y + wp["y"]
        target_z = wp.get("z", DEFAULT_ALTITUDE)
        speed    = wp.get("speed", DEFAULT_SPEED)
        desc     = wp.get("description", f"航点{idx+1}")

        print(f"[航点 {idx+1}/{len(waypoints)}] {desc}")
        print(f"  目标: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f}, 速度={speed} m/s")

        # 发送指令，最多重试3次
        reached = False
        for attempt in range(3):
            print(f"  [发送指令] moveToPositionAsync... (第{attempt+1}次)")
            client.moveToPositionAsync(
                target_x, target_y, target_z,
                velocity=speed,
                timeout_sec=60,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(False, 0)
            )

            reached = wait_until_reached(client, target_x, target_y, target_z)
            if reached:
                break
            if attempt < 2:
                print(f"  [重试] 第{attempt+1}次未到达，重新发送指令...")

        if not reached:
            print(f"  [放弃] 航点 {idx+1} 重试3次仍未到达，跳过")

        # 打印实际位置
        cur = client.getMultirotorState().kinematics_estimated.position
        print(f"  [实际位置] x={cur.x_val:.2f}, y={cur.y_val:.2f}, z={cur.z_val:.2f}")

        # 到达后悬停
        print(f"  [悬停] {HOVER_DURATION}s")
        client.hoverAsync()
        time.sleep(HOVER_DURATION)

    print(f"\n[执行] 所有航点完成，执行: {action_after}")
    if action_after == "land":
        print("[执行] 降落中...")
        client.landAsync().join()
        client.armDisarm(False)
    elif action_after == "return":
        print("[执行] 返回起点...")
        client.moveToPositionAsync(
            base_x, base_y, DEFAULT_ALTITUDE,
            velocity=DEFAULT_SPEED
        ).join()
        client.landAsync().join()
        client.armDisarm(False)
    else:
        print("[执行] 悬停")
        client.hoverAsync().join()