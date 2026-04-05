
import airsim
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from openai import OpenAI
from vln_demo.utils import get_scene_image, get_drone_state, parse_waypoints_from_vlm

QWEN_API_KEY  = os.environ["QWEN_VLM_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-vl-max"
INSTRUCT = "飞到前方路灯左侧0.1m处，降落"
# INSTRUCT = "飞到前方路灯右侧，悬停3秒后，继续前进至红绿灯口"
# INSTRUCT = "绕面前电线杆一圈后，停靠在第一个长椅上"


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
# client.hoverAsync().join()
# 等待停稳
time.sleep(2)


qwen_client = qwen_client = OpenAI(
                    api_key=QWEN_API_KEY,
                    base_url=QWEN_BASE_URL,
                    )  
_, img_base64 = get_scene_image(client)         
init_state = get_drone_state(client)
print(f"initial states are: {init_state}\n ")

plan = parse_waypoints_from_vlm(INSTRUCT, init_state, img_base64, qwen_client)
waypoints = plan.get("waypoints", [])
print(f" Total {len(waypoints)} points!\n")
print(f"Waypoints are: {waypoints}\n")

est_init_state = client.getMultirotorState()
base_pos = est_init_state.kinematics_estimated.position
base_x = base_pos.x_val
base_y = base_pos.y_val
base_z = base_pos.z_val
# print(f"estimated x: {base_x}\n")
# print(f"estimated y: {base_y}\n")
# print(f"estimated z: {base_z}\n")

prev_target_x = base_x
prev_target_y = base_y

speed = 1
for idx, wp in enumerate(waypoints):
    target_x = prev_target_x + wp["x"]
    target_y = prev_target_y + wp["y"]
    # target_z = base_z - wp["z"]

    desc = wp.get("description", f"航点{idx+1}")
    print(f"  目标: x={target_x:.1f}, y={target_y:.1f} \nfor purpose: {desc}")

    pre_state = client.getMultirotorState()
    pre_pos = pre_state.kinematics_estimated.position

    client.moveToPositionAsync(
                target_x, target_y, base_z,
                velocity=speed,
                timeout_sec=60,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
    
    curr_state = client.getMultirotorState()
    curr_pos = curr_state.kinematics_estimated.position
     
    actual_dx = curr_pos.x_val - pre_pos.x_val
    actual_dy = curr_pos.y_val - pre_pos.y_val
    print(f"  实际位移: dx={actual_dx:.3f}, dy={actual_dy:.3f}")
    print(f"  误差:     ex={actual_dx - wp['x']:.3f}, ey={actual_dy - wp['y']:.3f}")

    prev_target_x = target_x
    prev_target_y = target_y