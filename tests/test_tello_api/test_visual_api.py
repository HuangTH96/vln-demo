import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))    # 项目源目录 
from vln_demo.utils import apply_windows_conda_path_fix
apply_windows_conda_path_fix()

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