# runner.py
from platforms.airsim_platform import AirSimPlatform
from platforms.tello_platform  import TelloPlatform
# from platforms.offline_platform import OfflinePlatform
from backends.cloud_qwen  import CloudQwenBackend
from backends.local_qwen  import LocalQwenBackend
from algorithms.naive_vln import NaiveVLN
# from algorithms.spf_agent import SPFAgent

def build_system(cfg: dict):
    """工厂函数：根据配置组合三层"""

    # 1. 构建Platform
    platform_map = {
        "airsim":  AirSimPlatform,
        "tello":   TelloPlatform,
        # "offline": OfflinePlatform,
    }
    platform = platform_map[cfg["platform"]](cfg.get("platform_cfg", {}))

    # 2. 构建Backend
    backend_map = {
        "cloud_qwen": CloudQwenBackend,
        "local_qwen": LocalQwenBackend,
    }
    backend = backend_map[cfg["backend"]](cfg.get("backend_cfg", {}))

    # 3. 构建Algorithm（注入platform和backend）
    algo_map = {
        "naive": NaiveVLN,
        # "spf":   SPFAgent,
    }
    algorithm = algo_map[cfg["algorithm"]](
        backend  = backend,
        platform = platform,
        cfg      = cfg.get("algo_cfg", {}),
    )

    return algorithm


# ── 场景一：本地AirSim + Cloud API + Naive ──
agent = build_system({
    "platform": "airsim",
    "backend":  "cloud_qwen",
    "algorithm": "naive",
    "backend_cfg": {"api_key": "sk-xxx"},
    "algo_cfg":    {"platform_type": "airsim"},
})
agent.run("飞向前方红色建筑")

# ── 场景二：HPC离线评估 + Local模型 + SPF ──
agent = build_system({
    "platform": "offline",
    "backend":  "local_qwen",
    "algorithm": "spf",
    "platform_cfg": {"dataset_path": "/fs0/datasets/openfly"},
    "backend_cfg":  {"model_path": "/fs0/models/Qwen2-VL-7B"},
})
agent.run("Fly to the red building")
# TODO: 应该从数据集中读取指令和图像

# ── 场景三：Tello + Cloud API + Naive ──
agent = build_system({
    "platform": "tello",
    "backend":  "cloud_qwen",
    "algorithm": "naive",
    "algo_cfg":  {"platform_type": "tello"},
})
agent.run("向前飞行到走廊尽头")