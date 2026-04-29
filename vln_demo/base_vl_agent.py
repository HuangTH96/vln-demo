from abc import ABC, abstractmethod

# ============================================================
# 抽象基类：定义VLN Agent的完整接口
# ============================================================
class VLAgentBase(ABC):
    """
    VL Agent抽象基类。
    """

    def __init__(self, cfg: dict):
        """
        cfg: 配置字典，子类按需取用
        示例:
          cfg = {
              "image_size": (512, 512),
              "max_retries": 3,
              "action_type": ["semantic_action",        # 徐
                            "predicted_3D_waypoints",   # 黄    
                            "2D_pixel_coordinates"],    # SPF、Fly0
              ...
          }
        """
        self.cfg = cfg
        self.image_size  = cfg.get("image_size",  (512, 512))
        self.max_retries = cfg.get("max_retries", 3)
        self.action_type = cfg.get("action_type")

    @abstractmethod
    def _get_vl_client(self):
        """ 根据需要构建不同的VLM Agent """
        raise NotImplementedError
    
    @abstractmethod
    def _build_prompt(self):
        """
        构建多模态对话消息列表。
        """
        raise NotImplementedError

    @abstractmethod
    def _get_response(self, messages: list[dict]) -> str:
        """
        输入: build_prompt() 返回的消息列表
        输出: 模型原始文本响应（未经parse）
        """
        raise NotImplementedError
    
    @abstractmethod
    def _parse_response(self):
        """
        从模型原始输出中解析标准动作名称。
        容错处理各种输出格式。
        """
        raise NotImplementedError

    @abstractmethod
    def step(self):
        """
        完整的单步决策：instruct + img -> build_prompt → get_response → parse_response
        带重试机制。
        """
        raise NotImplementedError




