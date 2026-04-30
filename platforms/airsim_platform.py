import airsim
import cv2, base64
import numpy as np
import logging
from .base_platform import PlatformBase
import time

class AirSimPlatform(PlatformBase):

    def __init__(self, cfg: dict):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # 图像设置
        self.image_camera_id = cfg.get("image_camera_id", "0")
        self.image_type_id = cfg.get("image_type_id", "0")

        self.wps_keys = cfg.get("required_wps_keys", "{}")

    def get_image(self):
        png_image = self.client.simGetImage(self.image_camera_id, self.image_type_id)
        image_base64    = base64.b64encode(png_image).decode('utf-8')
        return image_base64
    
    def get_state(self):
        pass

    def rel2abs(self, waypoints: list, cur_position) -> list:
        """
        将 VLM 输出的相对航点转换为 AirSim 绝对坐标（以出生点为原点）
        
        current_pos: 本轮指令执行前无人机的当前位置
        """
        abs_wps = []
        for wp in waypoints:
            abs_wps.append({
                "x": wp["x"] + cur_position.x_val,
                "y": wp["y"] + cur_position.y_val,
                "z": wp["z"] + cur_position.z_val,
                "description": wp.get("description", "")
            })
        return abs_wps

    def wps2path(self, abs_wps):
        path = []
        for i, wp in enumerate(abs_wps):
            assert wp.keys() == self.wps_keys, \
                f"航点{i} 字段不匹配，期望： {self.wps_keys}, 实际：{set(wp.keys())}\n"

            logging.info(f"{i+1} abs_wps: \n-x: {wp['x']}\n-y: {wp['y']}\n-z: {wp['z']}\n{wp['description']}\n")
            
            path.append(airsim.Vector3r(
                wp["x"],
                wp["y"],
                wp["z"]
            ))
        return path

    def execute(self, abs_wps, speed: float):
        path = self._wps2path(abs_wps)
        self.client.moveOnPathAsync(path, speed).join()
    
    def get_position(self):
        return self.client.getMultirotorState().kinematics_estimated.position

    def takeoff(self) -> bool:
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            return True
        except Exception as e:
            logging.error(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        try:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            return True
        except Exception as e:
            logging.error(f"Land failed: {e}")
            return False

    def hover(self) -> bool:
        try:
            self.client.hoverAsync().join()
            return True
        except Exception as e:
            logging.error(f"Hover failed: {e}")
            return False
        
    # TODO: 放在这儿么？
    def _keepalive_thread(self):
        # TODO： 如何保证线程顺序？
        self.client.enableApiControl(False)
        time.sleep(0.5)

        self.client.enableApiControl(True)
        time.sleep(0.5)
        pass