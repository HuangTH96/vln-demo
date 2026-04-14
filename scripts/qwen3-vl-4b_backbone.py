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
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from typing import Tuple

# == config qwen3_vl_4b
MODEL_PATH = "/home/huangth/mLLM/qwen3-vl-4b"

DEFAULT_SPEED      = 3.0    # 默认飞行速度 m/s
DEFAULT_ALTITUDE   = -5.0   # 默认飞行高度（NED坐标，负值=向上）
HOVER_DURATION     = 3.0    # 到达航点后悬停时间 s
IMAGE_CAMERA_ID    = "0"    # AirSim 相机 ID

# == 加载本地 qwen3_vl_4b
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,   # 双重量化再省显存
    bnb_4bit_compute_dtype=torch.float16
    )
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# == utils: retrive image 
def get_scene_image(client: airsim.MultirotorClient) -> Tuple[np.ndarray, str]:
    """
    从 AirSim 获取当前场景图像
    返回: (numpy图像, base64字符串)
    """
    responses = client.simGetImages([
        airsim.ImageRequest(IMAGE_CAMERA_ID, airsim.ImageType.Scene, False, False)
    ])
    # response = responses[0]
    # img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    # img_rgb = img1d.reshape(response.height, response.width, 3)

    # # 编码为 base64 供 VLM 使用
    # _, buffer = cv2.imencode('.jpg', img_rgb)
    # img_base64 = base64.b64encode(buffer).decode('utf-8')

    # return img_rgb, img_base64

    response = responses[0]
    png_bytes = response.image_data_uint8
    img_base64 = base64.b64encode(png_bytes).decode('utf-8')
    return png_bytes, img_base64


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
        "altitude_m": round(-pos.z_val, 2),   # NED转高度
    }

# == High level planner: Qwen_vl_max
SYSTEM_PROMPT = """
你是一个无人机飞行控制助手。
用户会给你一段自然语言指令和当前无人机的状态信息，以及当前视角的图像。
你需要根据指令，生成一系列三维航点（waypoints）供无人机执行。

坐标系说明（NED坐标系）：
- X轴：向北为正（前方）
- Y轴：向东为正（右方）
- Z轴：向下为正（负值=向上，如 z=-5 表示高度5米）

输出格式要求（严格JSON，不要有任何多余文字）：
{
  "waypoints": [
    {"x": 1.0, "y": 0.0, "z": -5.0, "speed": 3.0, "description": "向前飞1米"},
    {"x": 2.0, "y": 0.0, "z": -5.0, "speed": 3.0, "description": "向前飞2米"},
    {"x": 3.0, "y": 0.0, "z": -5.0, "speed": 3.0, "description": "向前飞3米"},
    {"x": 4.0, "y": 0.0, "z": -5.0, "speed": 3.0, "description": "向前飞4米"},
    {"x": 5.0, "y": 0.0, "z": -5.0, "speed": 3.0, "description": "向前飞5米"}
  ],
  "action_after": "hover",
  "summary": "向前飞5米，每1米一个航点"
}

action_after 可选值：
- "hover"  : 完成后悬停
- "land"   : 完成后降落
- "return" : 完成后返回起点

注意：
- 航点坐标是相对于当前位置的累计偏移量（不是每段的增量）
- 速度单位为 m/s，建议范围 1~10
- 高度保持默认值除非用户明确要求改变高度
- 如果指令不清晰，生成一个悬停在原地的航点
- 航点必须足够密集，每段距离不超过 1 米，长距离飞行需拆分为多个小航点
- 曲线或转弯路径需要更多航点来平滑过渡，转弯处每隔 0.3 米设置一个航点
- 每个航点都需要有清晰的 description 说明当前段的动作
"""

# def parse_waypoints_from_vlm(
#     instruction: str,
#     drone_state: dict,
#     img_base64: str
# ) -> dict:
#     """
#     调用 Qwen-VL API，根据指令和图像生成航点
#     """
#     state_str = json.dumps(drone_state, ensure_ascii=False, indent=2)

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{img_base64}"
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": (
#                         f"当前无人机状态：\n{state_str}\n\n"
#                         f"飞行指令：{instruction}\n\n"
#                         f"请根据图像中的环境信息和无人机状态，生成航点列表。"
#                     )
#                 }
#             ]
#         }
#     ]

#     print(f"\n[VLM] 正在解析指令: '{instruction}'")

#     response = qwen_client.chat.completions.create(
#         model=QWEN_MODEL,
#         messages=messages,
#         max_tokens=1024,
#         temperature=0.1,   # 低温度保证输出稳定
#     )

#     raw = response.choices[0].message.content.strip()
#     print(f"[VLM] 原始响应:\n{raw}\n")

#     # 提取 JSON（防止模型输出多余文字）
#     start = raw.find('{')
#     end   = raw.rfind('}') + 1
#     if start == -1 or end == 0:
#         raise ValueError(f"VLM 输出中没有找到 JSON: {raw}")

#     result = json.loads(raw[start:end])
#     return result


def parse_waypoints_from_vlm(
    instruction: str,
    drone_state: dict,
    img_base64: str
) -> dict:
    """
    使用本地 Qwen3-VL 模型推理生成航点
    """
    state_str = json.dumps(drone_state, ensure_ascii=False, indent=2)

    # 构造消息（本地模型格式）
    content = []

    # 如果有图像则加入
    if img_base64:
        # base64 转为临时图像文件供 processor 使用
        import tempfile
        img_bytes = base64.b64decode(img_base64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes)
            tmp_img_path = f.name

        content.append({"type": "image", "image": f"file://{tmp_img_path}"})
    else:
        print(f"WARNING! No RT image captured!\n")

    content.append({
        "type": "text",
        "text": (
            f"当前无人机状态：\n{state_str}\n\n"
            f"飞行指令：{instruction}\n\n"
            f"请根据图像中的环境信息和无人机状态，生成航点列表。"
        )
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content}
    ]

    print(f"\n[VLM] 正在本地推理，指令: '{instruction}'")

    # 处理输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
        )

    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0].strip()

    # 清理临时文件
    if img_base64 and os.path.exists(tmp_img_path):
        os.unlink(tmp_img_path)

    print(f"[VLM] 原始响应:\n{raw}\n")

    # 提取 JSON
    start = raw.find('{')
    end   = raw.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f"VLM 输出中没有找到 JSON: {raw}")

    result = json.loads(raw[start:end])
    return result

# == Dummy controller with navive obstacle avoidance based on CV2,referred navigation.py in ~/AirSim/PythonClient/multirotor
# def check_obstacle_ahead(client: airsim.MultirotorClient) -> bool:
    """
    简单前向障碍物检测（来自 navigation.py 的逻辑）
    返回 True 表示前方有障碍
    """
    # try:
    #     result = client.simGetImage(IMAGE_CAMERA_ID, airsim.ImageType.DepthVis)
    #     if result is None or len(result) < 10:
    #         return False

    #     raw = np.frombuffer(result, np.uint8)
    #     png = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    #     if png is None:
    #         return False

    #     gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    #     top  = np.vsplit(gray, 2)[0]
    #     bands = np.hsplit(top, [50, 100, 150, 200])
    #     maxes = [np.max(x) for x in bands]
    #     current = 255 - maxes[2]   # 正前方距离估计

    #     return current < 20   # 距离小于阈值则有障碍
    # except Exception:
    #     return False


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

    # 获取当前位置作为偏移基准
    state    = client.getMultirotorState()
    base_pos = state.kinematics_estimated.position
    base_x   = base_pos.x_val
    base_y   = base_pos.y_val

    for idx, wp in enumerate(waypoints):
        # 航点坐标 = 当前位置 + 偏移
        target_x = base_x + wp["x"]
        target_y = base_y + wp["y"]
        target_z = wp.get("z", DEFAULT_ALTITUDE)
        speed    = wp.get("speed", DEFAULT_SPEED)
        desc     = wp.get("description", f"航点{idx+1}")

        print(f"[航点 {idx+1}/{len(waypoints)}] {desc}")
        print(f"  目标: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f}, 速度={speed} m/s")

        # 障碍物检测
        # if check_obstacle_ahead(client):
            # print(f"  [警告] 前方检测到障碍物，悬停等待...")
            # client.hoverAsync().join()
            # time.sleep(2.0)

        # 飞向航点
        client.moveToPositionAsync(
            target_x, target_y, target_z,
            velocity=speed,
            timeout_sec=60,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()

        # 到达后悬停片刻
        print(f"  [到达] 悬停 {HOVER_DURATION}s")
        client.hoverAsync()
        time.sleep(HOVER_DURATION)

    # 完成后动作
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


# == main 
def main():
    print("=" * 60)
    print("  VLM 无人机控制系统")
    print("  模型: Qwen-VL  |  仿真: AirSim + PX4")
    print("=" * 60)

    # 连接 AirSim
    print("\n[系统] 连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[系统] 连接成功")

    # 打印初始状态
    state = get_drone_state(client)
    print(f"[系统] 初始状态:\n{pprint.pformat(state)}")

    # 起飞
    input("\n按 Enter 键起飞...")
    print("[系统] 起飞中...")
    client.takeoffAsync().join()
    time.sleep(2.0)
    print("[系统] 起飞完成，当前高度:", get_drone_state(client)["altitude_m"], "m")

    # 主控制循环
    print("\n" + "=" * 60)
    print("  输入自然语言指令控制无人机")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'state' 查看当前状态")
    print("  输入 'land' 立即降落")
    print("=" * 60)

    while True:
        try:
            instruction = input("\n>>> 请输入飞行指令: ").strip()

            if not instruction:
                continue

            if instruction.lower() in ('quit', 'exit'):
                print("[系统] 退出，降落中...")
                client.landAsync().join()
                break

            if instruction.lower() == 'state':
                state = get_drone_state(client)
                print(pprint.pformat(state))
                continue

            if instruction.lower() == 'land':
                print("[系统] 立即降落...")
                client.landAsync().join()
                client.armDisarm(False)
                break

            if instruction.lower() == 'hover':
                client.hoverAsync()
                print("[系统] 悬停")
                continue

            # 获取当前图像和状态
            print("[系统] 获取当前视角图像...")
            try:
                # img_rgb, img_base64 = get_scene_image(client)
                # cv2.imshow("当前视角", img_rgb)
                # cv2.waitKey(1)
                _, img_base64 = get_scene_image(client)
            except Exception as e:
                # print(f"[警告] 获取图像失败: {e}，使用空白图像")
                # blank = np.zeros((144, 256, 3), dtype=np.uint8)
                # _, buf = cv2.imencode('.jpg', blank)
                # img_base64 = base64.b64encode(buf).decode('utf-8')
                print(f"[警告] 获取图像失败: {e}，跳过图像")
                img_base64 = ""

            drone_state = get_drone_state(client)

            # 调用 Qwen-VL 生成航点
            try:
                plan = parse_waypoints_from_vlm(instruction, drone_state, img_base64)
                print(f"[VLM] 生成航点计划:\n{json.dumps(plan, ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"[错误] VLM 调用失败: {e}")
                continue

            # 确认执行
            confirm = input("\n是否执行以上航点计划？(y/n): ").strip().lower()
            if confirm != 'y':
                print("[系统] 已取消")
                continue

            # 执行航点
            execute_waypoints(client, plan)

        except KeyboardInterrupt:
            print("\n[系统] 用户中断，降落中...")
            client.landAsync().join()
            break
        except Exception as e:
            print(f"[错误] {e}")
            client.hoverAsync()

    # 清理
    client.armDisarm(False)
    client.enableApiControl(False)
    cv2.destroyAllWindows()
    print("[系统] 已退出")


if __name__ == "__main__":
    main()

