import airsim
import math
import time
import base64
import json
import os
from openai import OpenAI
from typing import Tuple
import logging
import time

# config logger module
logging.basicConfig(level=logging.INFO)

# config airsim
DEFAULT_SPEED      = 3.0    # 默认飞行速度 m/s
DEFAULT_ALTITUDE   = -5.0   # 默认飞行高度（NED坐标，负值=向上）
HOVER_DURATION     = 3.0    # 到达航点后悬停时间 s
IMAGE_CAMERA_ID    = "0"    # AirSim 相机 ID

# == High level planner: Qwen_vl_max
# TODO: VLM更适合输出绝对坐标,也就是如果起步时是(0,0,-3)
# 那么前进5米,最好输出是(5,0,-3)
SYSTEM_PROMPT = """
你是一个无人机飞行控制助手。
用户会给你一段自然语言指令和当前无人机的状态信息，以及当前视角的图像。
你需要根据指令，生成一系列三维航点（waypoints）供无人机执行。

坐标系说明（AirSim NED坐标系）：
- X轴：正方向为前进方向
- Y轴：正方向为向右移动
- Z轴：负方向为向上（z=-5 表示在起点上方5米处，z=0 表示与起点同高）

航点坐标规则（重要）：
- 用户输入位置作为坐标原点，你输出的每个航点的坐标是相对于原点的绝对坐标
- 例如：从当前位置（用户输入：（0.0, 0.0,-3.0）出发，先上升3米后，前进5米，再继续前进4米，再向右1米：
    第1个航点: x=5.0, y=0.0, z=-3.0   （从当前位置前进5米，同时上升3米）
    第2个航点: x=9.0, y=0.0, z=-3.0   （从第1个航点再前进4米）
    第3个航点: x=9.0, y=1.0, z=-3.0   （从第2个航点再右移1米）

输出格式要求（严格JSON，不要有任何多余文字）：
{
  "waypoints": [
    {"x": 5.0, "y": 0.0, "z": -3.0, "description": "上升3米并前进5米"},
    {"x": 9.0, "y": 0.0, "z": -3.0,  "description": "继续前进4米，保持高度"},
    {"x": 9.0, "y": 1.0, "z": -3.0,  "description": "向右移动1米，保持高度"}
  ],
  "action_after": "hover",
  "summary": "上升3米，向前飞9米，再右移1米，3个航点"
}

action_after 可选值：
- "hover"  : 完成后悬停
- "land"   : 完成后降落
- "return" : 完成后返回起点(0, 0, 0)

航点规划原则：
- 转向或曲线路径时，增加中间航点保证路径平滑
- 每个航点需要有清晰的description说明该段动作
- 单段移动距离建议不超过10米
- 输出航点个数诗情况而定，不少于3个

注意：
- 严格返回JSON，不包含任何Markdown代码块标记（如 ```json）或解释性文字
- 如果指令不清晰，返回原地顺时针旋转一圈的航点序列
"""

# == utils: retrive image and convert to base64 for transmission
def get_scene_image(client: airsim.MultirotorClient) -> Tuple[bytes, str]:
    """
    从 AirSim 获取当前场景图像
    返回: (numpy图像, base64字符串)
    """
    # 返回 PNG 格式的原始字节数据（bytes）
    png_image = client.simGetImage(IMAGE_CAMERA_ID, airsim.ImageType.Scene)

    # PNG 字节直接 base64 编码
    img_base64 = base64.b64encode(png_image).decode('utf-8')

    return img_base64

# == utils: retrive drone states
def get_drone_state(client: airsim.MultirotorClient, diff_flatness_variable = False) -> dict:
    """ 获取无人机当前的位置、速度和3D姿态，或者4D微分平坦变量"""
    state = client.getMultirotorState()
    position   = state.kinematics_estimated.position

    return {
        "x": position.x_val,
        "y": position.y_val,
        "z": position.z_val
    }

def calling_qwen(
    instruction: str,
    drone_state: dict,
    img_base64: str,
    qwen_client,
    QWEN_MODEL="qwen-vl-max"
) -> dict:
    # 将字典格式的drone_state转换成格式化的字符串，拼进文本消息里
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
        logging.error("\n缺少图像输入!\n")

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
    
    logging.info(f"\nCalling qwen...\n")
    start = time.time()
    response = qwen_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.1,
    )

    time_consume = time.time() - start
    logging.info(f"\nQwen response in {time_consume:.2f}s! \nBuilding navigation plan from qwen response...!\n")
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
    
    response = calling_qwen(instruction, drone_state, img_base64, qwen_client)
    raw = response.choices[0].message.content.strip()

    start = raw.find('{')
    end   = raw.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f"VLM 输出中没有找到 JSON: {raw}")

    plan = json.loads(raw[start:end])
    return plan