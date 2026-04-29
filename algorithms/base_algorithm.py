from abc import ABC, abstractmethod
from typing import Optional

class AlgorithmBase(ABC):
    """
    决策逻辑抽象。
    持有 backend 和 platform 的引用，负责协调两者。
    """

    def __init__(
        self,
        backend,     # VLMBackendBase实例
        platform,    # PlatformBase实例
        cfg: dict,
    ):
        self.backend  = backend
        self.platform = platform
        self.cfg      = cfg

    @abstractmethod
    def build_messages(
        self,
        instruction: str,
        img_base64: str,
        state: dict,
        history: Optional[list] = None,
    ) -> list[dict]:
        """
        构建发给VLM的消息。
        不同算法prompt完全不同，必须各自实现。
        """
        raise NotImplementedError

    @abstractmethod
    def parse_action(
        self,
        raw_response: str,
        state: dict,
    ) -> dict:
        """
        解析VLM输出为统一action格式。
        不同算法输出格式不同，必须各自实现。
        """
        raise NotImplementedError

    def step(
        self,
        instruction: str,
        history: Optional[list] = None,
    ) -> dict:
        """
        模板方法：流程固定，细节交给子类。
        1. 从platform获取图像和状态
        2. 构建messages（子类实现）
        3. 调用backend获取响应
        4. 解析action（子类实现）
        5. 返回action（由runner决定是否执行）
        """
        img_bgr, img_b64 = self.platform.get_image()
        state            = self.platform.get_state()
        messages         = self.build_messages(
                               instruction, img_b64, state, history
                           )
        raw_response     = self.backend.query(messages)
        action           = self.parse_action(raw_response, state)
        return action

    def run(
        self,
        instruction: str,
        max_steps: int = 50,
    ) -> list[dict]:
        """完整导航循环"""
        history = []
        trajectory = []

        for step_idx in range(max_steps):
            action = self.step(instruction, history)
            trajectory.append(action)

            # 执行action
            success = self.platform.execute(action)

            # 更新历史
            history.append({
                "step":   step_idx,
                "action": action,
            })

            if action.get("type") == "stop" or not success:
                break

        return trajectory
