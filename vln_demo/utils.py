import logging
logging.basicConfig(level=logging.INFO)
import json
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))    # 项目源目录

from typing import Tuple

# ======== common =========
def build_prompt(
        instruction: str,
        img_base64: str,
        mode: str = "sim",  # airsim 或者 tello
):
    user_content = []

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
            f"飞行指令：{instruction}\n\n"
            f"请根据图像中的环境信息和无人机状态，生成航点列表。"
        )
    })

    if mode == "sim":
        system_prompt = SYSTEM_PROMPT_SIM
    elif mode == "tello":
        system_prompt = SYSTEM_PROMPT_TELLO
    else:
        raise ValueError(f"不支持的模式 '{mode}'，只支持 'sim' 和 'tello'")

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    
    return prompt

def get_response(
    prompt,
    qwen_client,
    qwen_model="qwen-vl-max"
) -> dict:
    logging.info(f"\nCalling qwen...\n")
    start = time.time()
    response = qwen_client.chat.completions.create(
        model=qwen_model,
        messages=prompt,
        max_tokens=1024,
        temperature=0.1,
    )

    time_consume = time.time() - start
    logging.info(f"\nQwen response in {time_consume:.2f}s! \nBuilding navigation plan from qwen response...!\n")
    return response

def parse_response(
    response
) -> dict:
    """
    调用 Qwen-VL API，根据指令、无人机状态和图像生成航点
    """
    raw = response.choices[0].message.content.strip()

    start = raw.find('{')
    end   = raw.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f"VLM 输出中没有找到 JSON: {raw}")

    plan = json.loads(raw[start:end])

    waypoints = plan.get("waypoints", [])
    if not waypoints:
        raise ValueError(f"VLM 没有输出航点！")

    speed = plan.get("speed", 0.0)

    return waypoints, speed

# ========== airsim =========
from config.conf import Config
import base64
import airsim 

def get_scene_image_sim(client: airsim.MultirotorClient) -> Tuple[bytes, str]:
    """
    从 AirSim 获取当前场景图像
    返回: (numpy图像, base64字符串)
    """
    # 返回 PNG 格式的原始字节数据（bytes）
    png_image = client.simGetImage(Config.IMAGE_CAMERA_ID, airsim.ImageType.Scene)

    # PNG 字节直接 base64 编码
    img_base64 = base64.b64encode(png_image).decode('utf-8')
    return img_base64

def wps2path(waypoints) -> list:
    """
    将 VLM 输出的 waypoints 转换为 AirSim 的 path（list of Vector3r）
    """
    print(f"\nTotal {len(waypoints)} points!\n")

    path = []
    for i, wp in enumerate(waypoints):
        assert wp.keys() == Config.REQUIRED_WAYPOINT_KEYS, \
            f"航点[{i}] 字段不匹配，期望: {Config.REQUIRED_WAYPOINT_KEYS}，实际: {set(wp.keys())}"
        
        path.append(airsim.Vector3r(
            wp["x"],
            wp["y"],
            wp["z"]
        ))

    print(f"Path is:\n{path}\n")
    return path

SYSTEM_PROMPT_SIM = """
你是一个无人机飞行控制助手。
用户会给你一段自然语言指令和当前视角的图像。
你需要根据指令，生成一系列三维航点（waypoints）供无人机执行。

坐标系说明（AirSim NED坐标系）：
- X轴：正方向为前进方向
- Y轴：正方向为向右移动
- Z轴：负方向为向上（z=-5 表示在起点上方5米处，z=0 表示与起点同高）

航点坐标规则（重要）：
- 你输出的每个航点，描述的是相对于当前位置的坐标，坐标系单位为米
- 例如：从当前位置出发，先上升3米后，前进5米，再继续前进4米，再向右1米：
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
- 输出航点中，忽略当前位置
"""

# ========== tello ========
import numpy as np
import cv2

def get_scene_image_tello(frame_reader):
    """
    获取 Tello 视频帧并转换为 base64 字符串
    Tello 视频流是基于 UDP 传输的，需要一段时间才能渲染出画面面
    """
    # frame_reader = tello.get_frame_read()
    start_time = time.time()
    
    while True:
        frame = frame_reader.frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 判定标准：帧不为空且不是全黑（平均像素值大于阈值）
        if frame_rgb is not None and np.mean(frame_rgb) > 5:
            _, buffer = cv2.imencode('.png', frame_rgb)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        
        # 防止死循环：超过 5 秒还没画面就报错
        if time.time() - start_time > 5:
            raise RuntimeError("无法获取 Tello 视频流，请检查电池和连接")
        
        time.sleep(0.1)

SYSTEM_PROMPT_TELLO = """
你是一个无人机飞行控制助手。
用户会给你一段自然语言指令和当前视角的图像。
你需要根据指令，生成一系列三维航点（waypoints）供无人机执行。

坐标系说明（Tello坐标系）：
- X轴：正方向为前进方向
- Y轴：正方向为向左移动
- Z轴：正方向为向上（z=50 表示上升50厘米，z=-50 表示下降50厘米）
- 单位：厘米（cm），每个航点表示相对于上一个航点的位移增量

航点坐标规则（重要）：
- 你输出的每个航点，描述的是相对于上一个航点的位移，不是累计坐标
- 例如：从当前位置出发，先上升50厘米后，前进1米，再继续前进80厘米，再向右20厘米：
    第1个航点: x=100, y=0,  z=50,  （上升50厘米并前进1米）
    第2个航点: x=80,  y=0,  z=0,   （继续前进80厘米）
    第3个航点: x=0,   y=-20, z=0,   （右移20厘米）

Tello 位移限制：
- 每个航点每轴的位移范围：-500 ~ 500 厘米
- 合法位移：三轴合位移必须大于 20 厘米（否则 Tello 会报错）

输出格式要求（严格JSON，不要有任何多余文字）：
{
  "waypoints": [
    {"x": 100, "y": 0,  "z": 50,  "description": "上升50厘米并前进1米"},
    {"x": 80,  "y": 0,  "z": 0,  "description": "继续前进80厘米"},
    {"x": 0,   "y": -20, "z": 0,  "description": "向右移动20厘米"}
  ],
  "speed": 30,
  "action_after": "hover",
  "summary": "上升50厘米并向前飞180厘米，再右移20厘米，4个航点"
}

speed 说明：
- 单位：厘米/秒（cm/s），范围 10 ~ 100
- 根据任务场景合理设置，室内建议 20~40，室外开阔场地可适当提高

action_after 可选值：
- "hover"  : 完成后悬停
- "land"   : 完成后降落
- "return" : 完成后返回起点(0, 0, 0)

航点规划原则：
- 每个航点为增量位移，不是累计坐标
- 转向或曲线路径时，增加中间航点保证路径平滑
- 单段位移建议不超过 200 厘米
- 每个航点需要有清晰的 description 说明该段动作

注意：
- 严格返回JSON，不包含任何Markdown代码块标记（如 ```json）或解释性文字
- 如果指令不清晰，返回原地顺时针旋转一圈的航点序列
- 输出航点中，忽略当前位置
"""