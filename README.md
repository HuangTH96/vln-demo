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