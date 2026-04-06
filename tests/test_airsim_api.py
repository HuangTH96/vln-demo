import airsim
import numpy as np
import base64
import cv2
import time

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
    
    print(f"\n############## After taking off, stablizing pos for 5s... ############ \n")
    time.sleep(5)
    
    print(f"Taking off introduces position error! \n \
                    x: {position_after.x_val - position.x_val}\n \
                    y: {position_after.y_val - position.y_val}\n ")
    return airsim_client
airsim_client = initialize()

def test_moveToPositionAsync():
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


def test_simGetImage():
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

# def reset_after_test(airsim_client):
#     airsim_client.moveToPositionAsync(
#         0, 0, -3,
#         velocity=3.0,
#         timeout_sec=30,
#         drivetrain=airsim.DrivetrainType.ForwardOnly,
#         yaw_mode=airsim.YawMode(False, 0)
#     ).join()
#     time.sleep(2)

# def take_one_step(airsim_client, curr_position, dx=5.0, dy=0.0, dz=0.0, vel=1.0):
#     """ 默认向前运动5m """

#     # 防止在采取动作之前，飞行器就已经发生漂移
#     curr_position_1 = airsim_client.getMultirotorState().kinematics_estimated.position
#     assert curr_position.x_val == curr_position_1.x_val, "Before taking step, x_position is changed!"
#     assert curr_position.y_val == curr_position_1.y_val, "Before taking step, y_position is changed!"
#     assert curr_position.z_val == curr_position_1.z_val, "Before taking step, z_position is changed!"
#     # print(f"\nBefore moving, position is:\n-x: {curr_position_1.x_val}\n-y: {curr_position_1.y_val}\n-z: {curr_position_1.z_val}\n")
#     # print(f"\nNeed to move\n-x:{dx},\n-y:{dy},\n-z:{dz}\n")
#     # print(f"Consider current position, move to\n-x:{curr_position.x_val + dx}\n-y:{curr_position.y_val + dy}\n-z:{curr_position.z_val + dz}\n")

#     airsim_client.moveToPositionAsync(
#         curr_position.x_val + dx,   # 查看 test_moveToPositionAsync 
#         curr_position.y_val + dy,
#         curr_position.z_val + dz,
#         velocity=vel,
#         timeout_sec=30,
#         drivetrain=airsim.DrivetrainType.ForwardOnly,
#         yaw_mode=airsim.YawMode(False, 0)
#     ).join()

#     # 查看PX4的执行效果
#     position = airsim_client.getMultirotorState().kinematics_estimated.position
#     # print(f"After moving, position is: \n-x: {position.x_val}\n-y: {position.y_val}\n-z: {position.z_val}\n")

#     return position