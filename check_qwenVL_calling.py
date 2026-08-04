import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))    # 项目源目录 
from vln_demo.utils import apply_windows_conda_path_fix
apply_windows_conda_path_fix()

import base64
import cv2
from openai import OpenAI
from vln_demo.utils import build_prompt, get_response, parse_response
from config.conf import Config

# ======== setup qwen model ========
qwen_client = OpenAI(
                    api_key=Config.QWEN_API_KEY,
                    base_url=Config.QWEN_BASE_URL,
                    )  

INIT_FRAME_PATH = os.path.join(os.path.dirname(__file__), "test_image.jpg")

test_img = cv2.imread(INIT_FRAME_PATH)
if test_img is None:
    raise RuntimeError(f"[ERROR] 本地 init_frame 读取失败，请检查文件: {INIT_FRAME_PATH}")
print(f"读取本地测试图像\n - type: {type(test_img)}, dtype: {test_img.dtype}, shape: {test_img.shape}")

    
success, buffer = cv2.imencode(".jpg", test_img)
if not success:
    raise RuntimeError("fail: jpg encoding")
test_img_base64 = base64.b64encode(buffer).decode('utf-8')
instruct = "飞到红色大门前"

prompt = build_prompt(instruct, test_img_base64, "tello")
response = get_response(prompt, qwen_client, Config.QWEN_MODEL)
waypoints, speed = parse_response(response)

for wp in waypoints:
    print(f" waypoint is: {wp}\n")

