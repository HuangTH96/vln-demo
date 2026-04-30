# 文件结构

```
vln/
├── platforms/
│   ├── base_platform.py       🔧 补充生命周期抽象方法
│   ├── airsim_platform.py     🔧 补充takeoff/land/hover/is_safe
│   └── tello_platform.py      🔧 补充takeoff/land/hover/is_safe
│
├── backends/
│   ├── base_backend.py
│   ├── cloud_qwen.py
│   ├── local_qwen.py
│   └── trainable_qwen.py
│
├── algorithms/
│   ├── base_algorithm.py
│   └── naive_vln.py
│
├── prompts/
│   ├── __init__.py
│   ├── airsim_prompts.py
│   └── tello_prompts.py
│
├── configs/
│   ├── airsim_cloud.yaml      ❌ 新建
│   └── tello_cloud.yaml       ❌ 新建
│
├── scripts/
│   ├── run_airsim.py          ❌ 新建（原airsim_qwen_api.py）
│   └── run_tello.py           ❌ 新建（原tello_qwen_api.py）
│
└── runner.py                  ❌ 新建
```


# 模块说明
1. Algorithm
- build prompt
- parse response
- main while loop

2. Platform
- get_scene_image_sim
- connection check
- takeoff\ land
- get_position: cur_position、after_move
- wpspath
- airsim.moveOnPathAsync

3. Backend
- build qwen_client
- get_response

4. utils
- Prompts

## 在airsim中复现交互式naive vln
1. `runner.py`
- 构建NaiveVLN实例，开启主循环

2. platform: `airsim_platform.py`
- 实现 连接airsim，控制起飞、降落等准备工作
- 实现 `get_scene_image_sim`
- 实现 `rel2abs`
- 实现 运动执行： `client.moveOnPathAsync`
- config:   `image_camera_id`, `image_type_id`, `required_wps_keys`

3. backend: `cloud_qwen.py`
- 实例化 qwen_client
- 实现 `get_response`
- config: `vlm_base_url`、`model_name`, `max_tokens`, `temperatures`, `max_retires`

4. algorithm: `naive_vln.py`
- 实现 `build_prompt`
- 实现 `parse_response`
- 实现 主循环
- config: `platform_type`, `action_type_id`

5. *TODO*：
- 如何处理交互式指令？如何处理评估数据集中的指令和图像
- 如何保活？也就是`airsim_qwen_api.py` 中的
```
client.enableApiControl(False)
time.sleep(0.5)
client.enableApiControl(True)
time.sleep(0.5)
```
