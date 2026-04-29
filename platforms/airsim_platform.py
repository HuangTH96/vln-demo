import airsim
import cv2, base64
import numpy as np
import logging
from .base_platform import PlatformBase

class AirSimPlatform(PlatformBase):

    def __init__(self, cfg: dict):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.image_size = cfg.get("image_size", (512, 512))

    def get_image(self) -> tuple[np.ndarray, str]:
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center",
                                airsim.ImageType.Scene, False, False)
        ])
        resp   = responses[0]
        img    = np.frombuffer(resp.image_data_uint8,
                               dtype=np.uint8).reshape(resp.height, resp.width, 3)
        img    = cv2.resize(img, self.image_size)
        _, buf = cv2.imencode(".jpg", img)
        b64    = base64.b64encode(buf).decode()
        return img, b64
    
    def get_state(self):
        pass

    def rel2abs(self, waypoints: list, cur_position) -> list:
        abs_wps = []
        for wp in waypoints:
            abs_wps.append({
                "x": wp["x"] + cur_position.x_val,
                "y": wp["y"] + cur_position.y_val,
                "z": wp["z"] + cur_position.z_val,
                "description": wp.get("description", "")
            })
        return abs_wps

    def execute_waypoints(self, waypoints: list, speed: float):
        path = [airsim.Vector3r(wp["x"], wp["y"], wp["z"])
                for wp in waypoints]
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