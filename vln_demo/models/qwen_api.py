import airsim
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from openai import OpenAI
from vln_demo.utils import get_scene_image, get_drone_state, parse_waypoints_from_vlm

import logging

# config logger module
logging.basicConfig(level=logging.INFO)

# config qwen model
QWEN_API_KEY  = os.environ["QWEN_VLM_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-vl-max"
INSTRUCT = "飞到面前路灯左侧，与路灯保持平行"
# INSTRUCT = "飞到前方路灯右侧，悬停3秒后，继续前进至红绿灯口"
# INSTRUCT = "绕面前电线杆一圈后，停靠在第一个长椅上"


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# take off and keep hover
client.takeoffAsync().join()
time.sleep(2)

# model parameter
diff_flatness_variable = True   # drone state

# create qwen_model for calling api
qwen_model = OpenAI(
                    api_key=QWEN_API_KEY,
                    base_url=QWEN_BASE_URL,
                    )  

# build input(current viewpoint and drone states) to qwen model
img_base64 = get_scene_image(client)         
init_state = get_drone_state(client, diff_flatness_variable)
# init_state = get_drone_state(client,True)
logging.info(f"\nInitial states are: {init_state}\n")

# retrive waypoints from qwen returns
plan = parse_waypoints_from_vlm(INSTRUCT, init_state, img_base64, qwen_model)
waypoints = plan.get("waypoints", [])
logging.info(f"\nTotal {len(waypoints)} points!\n")
logging.info(f"\nWaypoints are: {waypoints}\n")

est_init_state = client.getMultirotorState()
base_pos = est_init_state.kinematics_estimated.position
base_x = base_pos.x_val
base_y = base_pos.y_val
base_z = base_pos.z_val

prev_target_x = base_x
prev_target_y = base_y

speed = 1
for idx, wp in enumerate(waypoints):
    target_x = prev_target_x + wp["x"]
    target_y = prev_target_y + wp["y"]
    target_z = base_z + wp["z"]

    desc = wp.get("description", f"航点{idx+1}")
    logging.info(f"\n目标: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f} \nfor purpose: {desc}")

    curr_state = client.getMultirotorState()
    curr_pos = curr_state.kinematics_estimated.position

    client.moveToPositionAsync(
                target_x, target_y, target_z,
                velocity=speed,
                timeout_sec=60,
                drivetrain=airsim.DrivetrainType.ForwardOnly,   # always head movement direction
                yaw_mode=airsim.YawMode(False, 0)   # ignore mannual yaw setting under ForwardOnly mode
            ).join()

    # TODO: client.moveToPositionAsync process differential flaness outputs
    
    new_state = client.getMultirotorState()
    new_pos = new_state.kinematics_estimated.position
     
    actual_dx = new_pos.x_val - curr_pos.x_val
    actual_dy = new_pos.y_val - new_pos.y_val
    logging.info(f"\n实际位移: dx={actual_dx:.3f}, dy={actual_dy:.3f}")
    logging.info(f"\n误差:  ex={actual_dx - wp['x']:.3f}, ey={actual_dy - wp['y']:.3f}")

    prev_target_x = target_x
    prev_target_y = target_y
    prev_target_z = target_z