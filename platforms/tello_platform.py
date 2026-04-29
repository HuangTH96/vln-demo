# platforms/tello_platform.py
import cv2, base64
import numpy as np
from .base_platform import PlatformBase
from djitellopy import Tello
import logging

import threading
import numpy as np
import socket
import time

class TelloPlatform(PlatformBase):

    def __init__(self, cfg: dict):
        self.tello = Tello()
        self.tello.connect()
        self.image_size = cfg.get("image_size", (512, 512))

        self.stop_event = threading.Event()
        self.monitor = cfg.get("show_realtime_video", True)
        if self.monitor:
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.frame_reader = self.tello.get_frame_read()
        else:
            self.video_thread = None
            self.frame_reader = None

    def get_image(self) -> tuple[np.ndarray, str]:
        frame = self.tello.get_frame_read().frame
        img   = cv2.resize(frame, self.image_size)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode()
        return img, b64
    
    def get_state(self):
        pass

    def execute_waypoints(self, waypoints: list, speed: float):
        for wp in waypoints:
            self.tello.go_xyz_speed(
                int(wp["x"]),
                int(wp["y"]),
                int(wp["z"]),
                int(speed)
            )

    def takeoff(self) -> bool:
        try:
            self.tello.takeoff()
            return True
        except Exception as e:
            logging.error(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        try:
            self.tello.land()
            return True
        except Exception as e:
            logging.error(f"Land failed: {e}")
            return False

    def hover(self) -> bool:
        try:
            # Tello没有hover指令，发送极小速度模拟悬停
            self.tello.send_rc_control(0, 0, 0, 0)
            return True
        except Exception as e:
            logging.error(f"Hover failed: {e}")
            return False

    def _keepalive(self):
        """每10秒发一次保活指令，防止 Tello 因超时无指令自动降落"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not self.stop_event.is_set():
            try: 
                sock.sendto(b'command', ('192.168.10.1', 8889))
            except Exception:
                pass
            self.stop_event.wait(timeout=10)
        sock.close()

    def _video_on(self):
            while not self.stop_event.is_set():
                frame = self.frame_reader.frame
                if frame is not None:  
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Tello 视角", frame_rgb)
                # TODO：按 q 可从视频窗口触发退出（waitKey 必须与 imshow 在同一线程）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
            cv2.destroyAllWindows()

    def _wait_for_video(self, timeout=10):
        start = time.time()
        while True:
            frame = self.frame_reader.frame
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame is not None and np.mean(frame) > 5:
                print("视频流已就绪")
                return
            if time.time() - start > timeout:
                raise RuntimeError("视频流超时，请检查连接")
        time.sleep(0.1)
    