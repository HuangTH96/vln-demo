import airsim
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from vln_demo.utils import get_scene_image_sim, get_drone_state_sim, parse_waypoints_from_vlm, plan2path

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
img_base64 = get_scene_image_sim(client)         
init_state = get_drone_state_sim(client, diff_flatness_variable)
logging.info(f"\nInitial positions are: {init_state}\n")

# retrive waypoints from qwen returns
plan = parse_waypoints_from_vlm(INSTRUCT, init_state, img_base64, qwen_model)
path = plan2path(plan)

speed = 1
client.moveOnPathAsync(path, velocity=1).join()
after_move = client.getMultirotorState().kinematics_estimated.position
print(f"after movements, the positions are: \n-x:{after_move.x_val}\n-y:{after_move.y_val}\n-z:{after_move.z_val}")