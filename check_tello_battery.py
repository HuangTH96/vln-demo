# =====================================================================================
# 仅windows - vs code需要配置路径
# VS Code激活conda环境时只加了Python解释器的路径，没有完整初始化conda环境的所有PATH，需要手动配置
import os
os.environ["PATH"] = r"D:\anaconda\envs\tello\Library\bin" + ";" + os.environ["PATH"]
# ======================================================================================
from djitellopy import Tello

tello = Tello()
tello.connect()
print(f"current battery is: {tello.get_battery()}")