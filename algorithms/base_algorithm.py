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
    def process_command(self):
        """
        支持交互式指令或者固定指令（比如从数据集中提取，或终端直接输入）
        """
        raise NotImplementedError

    @abstractmethod
    def _build_messages(self):
        """
        构建发给VLM的消息。
        不同算法prompt完全不同，必须各自实现。
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_action(
        self,
        raw_response: str,
        state: dict,
    ) -> dict:
        """
        解析VLM输出为统一action格式。
        不同算法输出格式不同，必须各自实现。
        """
        raise NotImplementedError

    @abstractmethod
    def _step(self):
        """
        模板方法：流程固定，细节交给子类。
        1. 从platform获取图像和状态
        2. 构建messages（子类实现）
        3. 调用backend获取响应
        4. 解析action（子类实现）
        5. 返回action（由runner决定是否执行）
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """完整导航循环"""
        raise NotImplementedError
