import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))    # 项目源目录
from vln_demo.utils import apply_windows_conda_path_fix
apply_windows_conda_path_fix()

from djitellopy import Tello

tello = Tello()
tello.connect()
print(f"current battery is: {tello.get_battery()}")