import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # config airsim
    DEFAULT_SPEED      = 3.0    # 默认飞行速度 m/s
    DEFAULT_ALTITUDE   = -5.0   # 默认飞行高度（NED坐标，负值=向上）
    HOVER_DURATION     = 3.0    # 到达航点后悬停时间 s
    IMAGE_CAMERA_ID    = "0"    # AirSim 相机 ID

    REQUIRED_WAYPOINT_KEYS = {"x","y","z","description"}

    # qwen
    QWEN_API_KEY  = os.environ["QWEN_VLM_KEY"]
    QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL    = "qwen-vl-max"