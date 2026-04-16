# =====================================================================================
# 仅windows - vs code需要配置路径
# VS Code激活conda环境时只加了Python解释器的路径，没有完整初始化conda环境的所有PATH，需要手动配置
import os
os.environ["PATH"] = r"D:\anaconda\envs\tello\Library\bin" + ";" + os.environ["PATH"]
# ======================================================================================
from djitellopy import Tello
import time

tello = Tello()
tello.connect()
tello.takeoff()
print(f"当前电量：{tello.get_battery()}")

"""
测试go_xyz_speed

- 单位为cm/s
- x正方向为前进；y正方向为向右移动；z正方向向上
"""
# 以50cm/s的速度前进1m
print("前进1m...")
tello.go_xyz_speed(130, 0, 0, 50)
# 以50cm/s的速度向左移动70cm
print("向左移动30cm...")
tello.go_xyz_speed(0, 70, 0, 50)
# 以50cm/s的速度向上移动50cm
print("上升0.5m...")
tello.go_xyz_speed(0, 0, 50, 25)
time.sleep(5)
print("结束")
tello.land()
