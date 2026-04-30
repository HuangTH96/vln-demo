import airsim
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from openai import OpenAI
from vln_demo.utils import build_prompt, get_response, get_scene_image_sim, parse_response, rel2abs, wps2path
import logging
logging.basicConfig(level=logging.INFO)
from config.conf import Config

# connect to the AirSim simulator  -> class AirSimPlatform
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# take off and keep hover
client.takeoffAsync().join()
time.sleep(2)
cur_position = client.getMultirotorState().kinematics_estimated.position

# create qwen_model for calling api
qwen_client = OpenAI(
                    api_key=Config.QWEN_API_KEY,
                    base_url=Config.QWEN_BASE_URL,
                    )  

print("无人机已就绪...输入 q 退出")
while True:
    # 获取用户输入
    instruct = input("\n请输入指令：").strip()

    # 检查是否退出
    if instruct.lower() == 'q':
        print("退出指令模式，无人机降落中...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        break

    if not instruct:
        print("指令不能为空，请重新输入。")
        continue

    try:
        logging.info(f"正在执行指令：{instruct}")

        # build input(current viewpoint and drone states) to qwen model
        img_base64 = get_scene_image_sim(client)
        prompt = build_prompt(instruct, img_base64)
        response = get_response(prompt, qwen_client, Config.QWEN_MODEL)
        abs_waypoints, _ = parse_response(response, cur_position)
        path = wps2path(abs_waypoints)
        
        client.enableApiControl(False)
        time.sleep(0.5)
        client.enableApiControl(True)
        time.sleep(0.5)

        speed = 1.0
        client.moveOnPathAsync(path, velocity=speed).join()

        after_move = client.getMultirotorState().kinematics_estimated.position
        print(f"指令执行完成，当前位置：\n  x: {after_move.x_val:.2f}\n  y: {after_move.y_val:.2f}\n  z: {after_move.z_val:.2f}")
        print("请继续下达指令，或输入 q 退出。")
        cur_position = after_move

    except Exception as e:
        logging.error(f"执行指令时出错：{e}")
        print("指令执行失败，请检查指令或重试。")