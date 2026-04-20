# =====================================================================================
# 仅windows - vs code需要配置路径
# VS Code激活conda环境时只加了Python解释器的路径，没有完整初始化conda环境的所有PATH，需要手动配置
import os
os.environ["PATH"] = r"D:\anaconda\envs\tello\Library\bin" + ";" + os.environ["PATH"]
# ======================================================================================
from djitellopy import Tello
import time
import pytest 

@pytest.fixture(scope="module")
def tello():
    tello = Tello()
    tello.connect()
    print(f"Taking off in 3 seconds...")
    time.sleep(3)

    tello.takeoff()
    
    # pytest fixture的固定用法，把fixture分成前置和后置两个部分
    yield tello

    # 后置程序：测试结束后执行
    tello.land()

"""
测试go_xyz_speed

- 单位为cm/s
- x正方向为前进；y正方向为向右移动；z正方向向上
- 依赖 VPS（视觉定位系统），底部摄像头需要能看清地面纹理才能做位置估计，否则会出现error: no valid imu
"""
def test_go_xyz_speed(tello):
    # 以10cm/s的速度前进1m
    tello.go_xyz_speed(100, 0, 0, 80)
    # 以50cm/s的速度向左移动70cm
    tello.go_xyz_speed(0, 70, 0, 50)
    # 以50cm/s的速度向上移动50cm
    tello.go_xyz_speed(0, 0, 50, 25)
    # 回到起点
    tello.go_xyz_speed(-100, -70, -50, 80)
    time.sleep(3)
    
"""
测试 send_rc_control

- 四个参数分别为：左右移动速度、前后移动速度、上下移动速度、yaw_vel
- 向右 为正
- 向前 为正
- 向上 为正
- 顺时针旋转 为正
- 速度单位为cm/s，范围 -100 ~ 100

TODO:
- 是一个开环控制，控制精度极低
"""
def test_send_rc_control(tello):
    # 向右以0.5m/s的速度移动2s...
    tello.send_rc_control(50, 0, 0, 0)
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)

    # 向前以0.5m/s的速度移动2s...
    tello.send_rc_control(0, 50, 0, 0)
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)

    # 向上以0.5m/s的速度移动2s...  
    tello.send_rc_control(0, 0, 50, 0)
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)

    # 顺时针旋转90度/s...
    # TODO: 使用tello.rotate_clockwise()和tello.rotate_counter_clockwise()控制yaw角
    tello.send_rc_control(0, 0, 0, 45)
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)

    # 回到起点，恢复初始yaw角...
    tello.send_rc_control(-50, -35, -25, -45)
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)

    time.sleep(3)

"""
测试Tello状态相关

[1] get_current_state

- 左倾 为负
- 前倾 为负
- 左转 为负

[2] get_pitch

[3] get_roll    # TODO: roll值不准，需要进一步检测

[4] get_yaw

[5] query_distance_tof
"""
def test_get_current_state(tello):
    state = tello.get_current_state()
    print(f"当前状态: {state}")
    assert isinstance(state, dict)

    # 测试 fields name
    required_keys = [
        'pitch', 'roll', 'yaw',
        'vgx', 'vgy', 'vgz',
        'templ', 'temph',
        'tof', 'h', 'bat', 'baro',
        'time', 'agx', 'agy', 'agz',
        'received_at',
    ]
    state = tello.get_current_state()
    for key in required_keys:
        assert key in state, f"状态字典缺少字段: {key}"

    # 测试 主要fields 取值范围
    assert isinstance(state['pitch'], int)
    assert -90 <= state['pitch'] <= 90

    assert isinstance(state['roll'], int)
    assert -180 <= state['roll']  <= 180
    
    assert isinstance(state['yaw'], int)
    assert -180 <= state['yaw']   <= 180

    assert  30 <= state['tof'] <= 1000      # [cm] -> tello.query_distance_tof 激光测距，离地面高度
    assert  0 <= state['h'] <= 1000         # [cm] -> tello.get_height，以起飞点为基准做了归零的相对高度，会因为气压计漂移而归零
    assert  0 <= state['baro'] <= 10000     # [m] 气压计测量的，相对海拔基准的绝对气压高度，无飞行意义
    
    assert isinstance(state['vgx'], int)
    assert isinstance(state['vgy'], int)
    assert isinstance(state['vgz'], int)

    assert isinstance(state['agx'], float)
    assert isinstance(state['agy'], float)
    assert isinstance(state['agz'], float)

def test_get_pitch(tello):
    pitch = tello.get_pitch()
    assert -180 <= pitch <= 180

def test_get_roll(tello):
    roll = tello.get_roll()
    assert -180 <= roll <= 180

def test_get_yaw(tello):
    yaw = tello.get_yaw()
    assert -180 <= yaw <= 180

def test_query_distance_tof(tello):
    dist = tello.query_distance_tof()
    assert isinstance(dist, (int, float))
    assert 0 <= dist <= 1000
    print(f"当前距离: {dist} cm")
