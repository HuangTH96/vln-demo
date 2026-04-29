from abc import ABC, abstractmethod
import numpy as np

class PlatformBase(ABC):
    """感知 + 执行，与算法和模型无关"""

    @abstractmethod
    def get_image(self) -> tuple[np.ndarray, str]:
        """返回 (bgr_array, base64_string)"""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        """
        返回平台当前状态，统一格式：
        {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading":  0.0,    # 偏航角（度）
            "speed":    0.0,
        }
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, action: dict) -> bool:
        """
        执行算法输出的action，统一格式：
        {
            "type": "waypoints",          # 或 "semantic" / "pixel"
            "data": [...],
            "speed": 1.0
        }
        返回是否执行成功
        """
        raise NotImplementedError
    
    @abstractmethod
    def takeoff(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def land(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def hover(self) -> bool:
        """悬停/急停，紧急情况用"""
        raise NotImplementedError