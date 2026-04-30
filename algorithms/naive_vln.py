import json
from typing import Optional
from .base_algorithm import AlgorithmBase
from platforms import AirSimPlatform, TelloPlatform
from prompts.airsim_prompts import SYSTEM_PROMPT_SIM, SYSTEM_PROMPT_TELLO
import logging

class NaiveVLN(AlgorithmBase):
    """
    原始实现：VLM直接输出3D waypoints
    """
    def __init__(
            self,
            backend,
            platform,
            cfg: dict
    ):
        super.__init__(
            self,
            backend, 
            platform,
            cfg
        )

        self.action_type = cfg.get("action_type_id", 0)

    def build_messages(
        self,
        instruction: str,
        img_base64: str,
        state: dict = None,
        history: Optional[list] = None,
    ) -> list[dict]:
        # 根据platform类型选择prompt
        platform_type = self.cfg.get("platform_type", "airsim")
        system_prompt = (SYSTEM_PROMPT_SIM
                         if platform_type == "airsim"
                         else SYSTEM_PROMPT_TELLO)

        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            },
            {
                "type": "text",
                "text": (
                    f"飞行指令：{instruction}\n"
                    f"当前位置：{state['position']}\n"
                    f"请生成航点列表。"
                )
            }
        ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

    def _parse_action(self, vlm_response: str, cur_position) -> dict:
        start = vlm_response.find('{')
        end   = vlm_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON in response: {vlm_response}")

        plan      = json.loads(vlm_response[start:end])
        waypoints = plan.get("waypoints", [])
        speed     = plan.get("speed", 1.0)

        if not waypoints:
            raise ValueError("No waypoints in response")

        if isinstance(self.platform, AirSimPlatform):
            # 相对→绝对坐标（调用platform的坐标转换）
            abs_wps = self.platform.rel2abs(waypoints, cur_position)
            action = self.platform.wps2path(abs_wps)
        elif isinstance(self.platform, TelloPlatform):
            action = waypoints
        else:
            raise ValueError("当前只支持AirSim和Tello，不支持{self.platform}!\n")

        return {
            "action_type":  self.action_type,
            "data":      action,
            "speed":     speed,
        }
    
    def _update_position(self):
        self.cur_position = self.platform.get_position()

    def _step(
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
        img_b64 = self.platform.get_image()
        # png_imamge = self.platform.simGetImage(self.cfg.get("image_camera_id", "0"), self.platform.ImageType.Scene)
        # cur_position = self.platform.get_position()
        messages = self.build_messages(instruction, img_b64)
        response = self.backend.query(messages)
        action = self._parse_action(response, self.cur_position)
        return action

    def get_state(self):
        return None
    
    # TODO: 如何实现交互式指令    
    def run(
        self,
        instruction: str,
        max_steps: int = 50,
    ) -> list[dict]:
        """完整导航循环"""

        # 准备工作：起飞并悬停
        self.platform.takeoff()

        # cur_position = self.platform.get_position()

        history = []
        trajectory = []
        # for step_idx in range(max_steps):
        step_idx = 0
        mission_done = False
        while step_idx <= max_steps and not mission_done:
            if instruction.lower() == 'q':
                print("退出指令模式，无人机降落中...")
                mission_done = self.platform.land()
                
            if not instruction:
                print("指令不能为空，请重新输入...\n")
                continue
            
            logging.info(f"正在执行指令：{instruction}\n")

            action = self._step(instruction, history)
            trajectory.append(action)

            # 执行action
            success = self.platform.execute(action)

            # 更新
            self._update_position()
            history.append({
                "step":   step_idx,
                "action": action,
            })

            if action.get("type") == "stop" or not success:
                break

        return trajectory