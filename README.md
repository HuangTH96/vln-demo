# 分支说明
本分支通过抽象基类统一多平台接口（AirSim / Tello），实现算法与平台的解耦，支持在仿真和实机环境下对多种导航算法进行对比评估。该分支最终应该merge airsim-qwen和tello-qwen，模块化地复现：

1. naive vln-demo：`airsim_qwen_api.py` 和 `tello_qwen_api.py`
2. spf: `airsim_spf.py` 和 `tello_spf.py`

并根据fly0的思路评估这两个算法

# 工作记录
1. 实现 naive vln in airsim【完成】
2. 实现 naive vln with tello 【完成】
3. 实现 airsim spf
4. 实现 tello spf
5. 按照 fly0 评估 airsim spf
6. 按照 fly0 评估 tello spf
7. 设计系统级算法开发模式【抽象接口设计（tello、airsim、PX4、ROS）、各后端实现、坐标系转换、工厂模式】
