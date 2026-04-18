"""
    该文件主要实现让tello去跟踪VLN算法输出的waypoints，实现导航功能

    方案一：go_xyz_speed
    go_xyz_speed是阻塞式的，每个指令执行完才执行下一个，所以在每个waypoint之间会有明显的停顿和减速再加速。

    方案二：send_rc_control
    需要手动实现控制器：根据当前位置和目标位置计算每个方向的速度值，类似PID。问题是Tello没有GPS，无法确定自身位置
"""

# =====================================================================================
# 仅windows - vs code需要配置路径
# VS Code激活conda环境时只加了Python解释器的路径，没有完整初始化conda环境的所有PATH，需要手动配置
import os
os.environ["PATH"] = r"D:\anaconda\envs\tello\Library\bin" + ";" + os.environ["PATH"]
# ======================================================================================

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from openai import OpenAI
from vln_demo.utils import build_prompt, get_response, parse_response, get_scene_image_tello
from djitellopy import Tello
from config.conf import Config

import cv2
import threading
import time
import numpy as np

qwen_client = OpenAI(
                    api_key=Config.QWEN_API_KEY,
                    base_url=Config.QWEN_BASE_URL,
                    )  

# ======== initialize tello ========
tello = Tello()
tello.connect()
print(f"电池电量：{tello.get_battery()}%")
tello.streamon()
# tello.takeoff()

# ========= 等待视频流稳定 ========
# 顶层创建唯一 frame_reader，视频线程和截图函数共用，避免重复创建读取器
frame_reader = tello.get_frame_read()

def wait_for_video(timeout=10):
    """阻塞直到拿到有效视频帧，超时则报错"""
    print("正在等待视频流...")
    start = time.time()
    while True:
        frame = frame_reader.frame
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame is not None and np.mean(frame) > 5:
            print("视频流已就绪")
            return
        if time.time() - start > timeout:
            raise RuntimeError("视频流超时，请检查连接")
        time.sleep(0.1)

# ======== 后台视频线程 =========
stop_event = threading.Event()

def video_loop():
    while not stop_event.is_set():
        frame = frame_reader.frame
        if frame is not None:  
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Tello 视角", frame_rgb)
        # TODO：按 q 可从视频窗口触发退出（waitKey 必须与 imshow 在同一线程）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

# ========= 保活线程 =========
import socket
def keepalive_loop():
    """每10秒发一次保活指令，防止 Tello 因超时无指令自动降落"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not stop_event.is_set():
        try: 
            sock.sendto(b'command', ('192.168.10.1', 8889))
        except Exception:
            pass
        stop_event.wait(timeout=10)
    sock.close()

# ========= 先启动视频，确认后起飞 ========
wait_for_video()

video_thread = threading.Thread(target=video_loop, daemon=True)
video_thread.start()

input("按 Enter 键起飞！\n")
tello.takeoff()

keepalive_thread = threading.Thread(target=keepalive_loop, daemon=True)
keepalive_thread.start()

# ========== 主循环 ===========
print("Tello 已就绪，输入 q 降落")
while not stop_event.is_set():  # 支持从视频窗口出发退出
    # ======== 获取用户指令 ========
    instruct = input("\n请输入指令：").strip()

    if instruct.lower() == 'q' or stop_event.is_set():
        stop_event.set()

        print("Tello 降落中...")
        tello.land()
        tello.streamoff()
        tello.end()
        break

    if not instruct:
        print("指令不能为空，请重新输入。")
        continue

    try:
        print(f"正在执行指令：{instruct}")

        # ======== build input to qwen_vl ========
        img_base64 = get_scene_image_tello(frame_reader)
        prompt = build_prompt(instruct, img_base64, mode="tello")
        response = get_response(prompt, qwen_client, Config.QWEN_MODEL)
        waypoints, speed = parse_response(response)

        # ======== execution ========
        # safe_speed = max(10, min(int(speed * 100), 100))  # Tello 速度单位为 cm/s，范围 10~100
        safe_speed = 50
        for wp in waypoints:
            tello.go_xyz_speed(
                int(wp["x"]),
                int(wp["y"]),
                int(wp["z"]),
                safe_speed
            )
            print(f"前往：\n-x:{wp['x']}\n-y:{wp['y']}\n-z:{wp['z']}\n")

        battery = tello.get_battery()
        print(f"指令执行完成，当前电池电量：{battery}%")

        if battery < 10:
            print("WARNING: 电量不足 10%，3秒后自动降落！")
            stop_event.set()  # 关闭视频窗口
            time.sleep(3)
            tello.land()
            tello.streamoff()
            tello.end()
            break
        
        print("请继续下达指令，或输入 q 退出。")

    except Exception as e:
        print(f"执行指令时出错：{e}")
        print("指令执行失败，请检查指令或重试。")






