import json
from typing import Optional
from .base_algorithm import AlgorithmBase
from prompts.airsim_prompts import SYSTEM_PROMPT_SIM, SYSTEM_PROMPT_TELLO

class NaiveVLN(AlgorithmBase):
    """
    原始实现：VLM直接输出3D waypoints
    """

    def build_messages(
        self,
        instruction: str,
        img_base64: str,
        state: dict,
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

    def parse_action(self, raw_response: str, state: dict) -> dict:
        start = raw_response.find('{')
        end   = raw_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON in response: {raw_response}")

        plan      = json.loads(raw_response[start:end])
        waypoints = plan.get("waypoints", [])
        speed     = plan.get("speed", 1.0)

        if not waypoints:
            raise ValueError("No waypoints in response")

        # 相对→绝对坐标（调用platform的坐标转换）
        abs_wps = self.platform.rel2abs(waypoints, state["position"])

        return {
            "type":      "waypoints",
            "data":      abs_wps,
            "speed":     speed,
        }