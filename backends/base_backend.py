from abc import ABC, abstractmethod

class VLMBackendBase(ABC):
    """VLM调用抽象，与平台和算法无关"""

    @abstractmethod
    def query(
        self,
        messages: list[dict],   # 由Algorithm层构建好传入
        **kwargs,
    ) -> str:
        """
        输入：标准消息列表
        输出：模型原始文本响应
        Backend只管"怎么调用模型"，不管消息内容
        """
        raise NotImplementedError