# 仅windows - vs code需要配置路径
# VS Code激活conda环境时只加了Python解释器的路径，没有完整初始化conda环境的所有PATH，需要手动配置
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))     
from vln_demo.utils import apply_windows_conda_path_fix 
apply_windows_conda_path_fix()  # 修复windows下conda环境的PATH问题

from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

while True:
    img = frame_read.frame
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Tello输出RGB格式，cv显示BGR格式
    
    cv2.imshow("Tello Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()