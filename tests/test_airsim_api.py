import airsim
import numpy as np
import base64
import cv2
import time
import warnings
import pytest

def initialize():
    """
    初始化airsim_client, 起飞后悬停，等待指令
    """
    airsim_client = airsim.MultirotorClient()
    airsim_client.confirmConnection()
    airsim_client.enableApiControl(True)
    airsim_client.armDisarm(True)

    # record initial position
    position = airsim_client.getMultirotorState().kinematics_estimated.position
    print(f"Initial position is: \n-x: {position.x_val}\n-y: {position.y_val}\n-z: {position.z_val}\n")
    
    airsim_client.takeoffAsync().join()

    # record position after taking off
    position_after = airsim_client.getMultirotorState().kinematics_estimated.position
    print(f"After taking off, position is: \n-x: {position_after.x_val}\n-y: {position_after.y_val}\n-z: {position_after.z_val}\n")
    # time.sleep(3)
    
    # print(f"Taking off introduces position error in x and y direction! \n \
    #                 x: {position_after.x_val - position.x_val}\n \
    #                 y: {position_after.y_val - position.y_val}\n ")

    prepare_time = 3
    print(f"{prepare_time}秒后，即将执行测试任务...\n")
    time.sleep(prepare_time)
    
    return airsim_client

@pytest.fixture(scope="module")
def airsim_client():
    return initialize()

def test_moveToPositionAsync(airsim_client):
    """
    函数特点：
    
    - 接受三个方向的位移距离（米），以当前位置为基准，因此，需要传入 prev_position + desired_movement
    - 连续动作时，两次 moveToPositionAsync 之间不可以间隔超过0.5s,否则第二次指令不会被执行
    - 位移误差会随着运动次数累积
    """
    tolerance_step = 0.5
    tolerance_total = 1.0
    # reset_after_test(airsim_client)

    init_position = airsim_client.getMultirotorState().kinematics_estimated.position

    # first movement
    dx1 = 5.0
    dy1 = 0.0
    dz1 = -5.0

    # 检查起飞后的位置，明确moveToPositionAsync函数接受的是 init_position + desired_movement
    # print(f"\nBefore taking first movement, position is:\n-x: {init_position.x_val}\n-y: {init_position.y_val}\n-z: {init_position.z_val}\n")
    # print(f"\nNeed to move\n-x:{dx1},\n-y:{dy1},\n-z:{dz1}\n")
    # print(f"Consider current position, move to\n-x:{init_position.x_val + dx1}\n-y:{init_position.y_val + dy1}\n-z:{init_position.z_val + dz1}\n")

    airsim_client.moveToPositionAsync(
        init_position.x_val + dx1, 
        init_position.y_val + dy1,
        init_position.z_val + dz1,
        velocity=1.0,
        timeout_sec=30,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0)
    ).join()

    position1 = airsim_client.getMultirotorState().kinematics_estimated.position
    # 检查是否运动到期望位置。 TODO：是运动存在误差还是估计不准确？
    # print(f"\nAfter first step, current position is:\n-x: {position1.x_val}\n-y: {position1.y_val}\n-z: {position1.z_val}\n")
    assert abs((position1.x_val - init_position.x_val) - dx1) < tolerance_step
    assert abs((position1.y_val - init_position.y_val) - dy1) < tolerance_step
    assert abs((position1.z_val - init_position.z_val) - dz1) < tolerance_step

    # second movement
    dx2 = 5.0
    dy2 = 0.0
    dz2 = -4.0

    # 检查连续动作
    print(f"\nTaking second step, need to move\n-x:{dx2},\n-y:{dy2},\n-z:{dz2}\n")
    print(f"Consider current position, move to\n-x:{position1.x_val + dx2}\n-y:{position1.y_val + dy2}\n-z:{position1.z_val + dz2}\n")

    airsim_client.moveToPositionAsync(
        position1.x_val + dx2,  
        position1.y_val + dy2,
        position1.z_val + dz2,
        velocity=1.0,
        timeout_sec=30,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0)
    ).join()

    position2 = airsim_client.getMultirotorState().kinematics_estimated.position
    assert abs((position2.x_val - position1.x_val) - dx2) < tolerance_step
    assert abs((position2.y_val - position1.y_val) - dy2) < tolerance_step
    assert abs((position2.z_val - position1.z_val) - dz2) < tolerance_step

    # 检查误差是否会累积
    assert abs((position2.x_val - init_position.x_val) - (dx1 + dx2)) < tolerance_total
    assert abs((position2.y_val - init_position.y_val) - (dy1 + dy2)) < tolerance_total
    assert abs((position2.z_val - init_position.z_val) - (dz1 + dz2)) < tolerance_total

def test_simGetImage(airsim_client):
    """
    函数特征：

    - 返回bytes格式数据
    - 可以转变成numpy、png、base64
    """
    # 返回的是bytes
    png_bytes = airsim_client.simGetImage("0", airsim.ImageType.Scene)
    assert isinstance(png_bytes, bytes)

    # bytes可以转变成 numpy 数组
    png_np = np.frombuffer(png_bytes, dtype=np.uint8)
    png_np = cv2.imdecode(png_np, cv2.IMREAD_COLOR)
    assert isinstance(png_np, np.ndarray)

    # bytes 可以转变成base64 用于传输
    png_url = base64.b64encode(png_bytes).decode('utf-8')
    assert isinstance(png_url, str)

def test_moveOnPathAsync(airsim_client):
    """
    函数特点：

    - 将**出生点**作为原点，所有的waypoints都是相对于原点的绝对坐标
    - 连续动作时，需要重新授权ApiControl
    - 根据AirSim cpp源码可知，path是list of Vector3r
    """
    # init_position = airsim_client.getMultirotorState().kinematics_estimated.position
    # print(f"initial positions are: \n-x:{init_position.x_val}\n-y:{init_position.y_val}\n-z:{init_position.z_val}")
    
    # 该path是一个闭环，运动完后应该能回到第一个waypoint处
    path = [
        airsim.Vector3r(0,0,0),     # 因为是相对于出生点运动，所以这个路径会从起飞位置回到出生位置，
        airsim.Vector3r(0,0,-3),    # 返回到起飞后的位置

        airsim.Vector3r(3,0,-3),    # 向前移动3m
        airsim.Vector3r(3,-3,-3),   # 再向左移动3m
        airsim.Vector3r(3,-3,-5),   # 再升高2m
        airsim.Vector3r(0,-3,-3),    # 飞到相对于起始位置左侧3m处
    ]

    airsim_client.moveOnPathAsync(path, velocity=1).join()  # 必须加.join()，否则还没等运动结束，就得到after_move坐标
    after_move = airsim_client.getMultirotorState().kinematics_estimated.position
    print(f"after movements, the positions are: \n-x:{after_move.x_val}\n-y:{after_move.y_val}\n-z:{after_move.z_val}\n")

    position = airsim_client.getMultirotorState().kinematics_estimated.position

    tol = 0.5
    if abs(position.x_val - 5.0) >= tol:
        warnings.warn(f"x方向误差过大，超过阈值{tol}\n")

    if abs(position.y_val - 5.0) >= tol:
        warnings.warn(f"y方向误差过大，超过阈值{tol}\n")

    if abs(position.z_val - 5.0) >= tol:
        warnings.warn(f"z方向误差过大，超过阈值{tol}\n")
    
    print(f"After first round, uav's position is: \n-x:{position.x_val}\n-y:{position.y_val}\n-z:{position.z_val}\n")

    # 连续调用moveOnPathAsync需要重置ApiControl权限
    airsim_client.enableApiControl(False)
    time.sleep(0.5)
    airsim_client.enableApiControl(True)
    time.sleep(0.5)

    print("Implementing the second round of moveOnPathAsync...\n")
    # 检测是否能够连续执行
    # 第二轮仍然相对于出生位置运动，仍然是绝对坐标？
    path_2 = [
        airsim.Vector3r(0,0,-6),    # 既不是原地上升6米，也不是到起飞位置上方6米，而是到出生位置上方6米,也就是起飞位置上方3米   
    ]
    airsim_client.moveOnPathAsync(path_2, velocity=1).join()
    after_move_2 = airsim_client.getMultirotorState().kinematics_estimated.position
    print(f"after movements, the positions are: \n-x:{after_move_2.x_val}\n-y:{after_move_2.y_val}\n-z:{after_move_2.z_val}")
