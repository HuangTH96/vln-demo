from .base_vl_agent import VLAgentBase
from openai import OpenAI
import os
import time
import logging
logging.baseicConfig(level=logging.INFO)
import json
from typing import Optional
from typing import Optional
from .utils_naive_vln import *

class CloudQwenVL(VLAgentBase):
    """
    通过阿里云DashScope API调用 Qwen-VL。
    对应原来的 airsim_qwen_api.py 逻辑。
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.qwen_client = self._get_vl_client(cfg)
        self.model_name = cfg.get("model_name", "qwen-vl-max")
    
    def _get_vl_client(self, cfg):
        qwen_client = OpenAI(
            api_key=cfg.get("api_key") or os.environ["QWEN_API_KEY"],
            base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        return qwen_client
    
    def _build_prompt(
            self,
            instruction: str,
            img_base64,
            history: Optional[list[dict]] = None,
            extra_context: Optional[str] = None,
            mode: str = "airsim"     # OR tello
    ):
        user_content = []

        if img_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
        else:
            raise ValueError("\n缺少图像输入!\n")

        # TODO: 丰富prompt，比如加入历史信息和其他额外信息
        user_content.append({
            "type": "text",
            "text": (
                f"飞行指令：{instruction}\n\n"
                f"请根据图像中的环境信息和无人机状态，生成航点列表。"
            )
        })

        if mode == "airsim":
            system_prompt = SYSTEM_PROMPT_SIM
        elif mode == "tello":
            system_prompt = SYSTEM_PROMPT_TELLO
        else:
            raise ValueError(f"不支持的模式 '{mode}'，只支持 'sim' 和 'tello'")

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        return prompt
    
    def _get_response(
            self,
            prompt: str,
    ):
        start = time.time()
        response = self.qwen_client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=1024,
            temperature=0.1
        )

        time_consume = time.time() - start
        logging.info(f"\nQwen response in {time_consume:.2f}s! \nBuilding navigation plan from qwen response...!\n")
        return response
    
    def _parse_response(
            self,
            response,
            cur_position        
    ):
        raw = response.choices[0].message.content.strip()

        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"VLM 输出中没有找到 JSON: {raw}")

        plan = json.loads(raw[start:end])

        waypoints = plan.get("waypoints", [])
        if not waypoints:
            raise ValueError(f"VLM 没有输出航点！")
        
        # TODO: tello不需要
        abs_waypoints = self._rel2abs(waypoints, cur_position)
        speed = plan.get("speed", 0.0)

        return abs_waypoints, speed
    
    def _rel2abs(self, wps, cur_position):
        """
        将 VLM 输出的相对航点转换为 AirSim 绝对坐标（以出生点为原点）
        
        current_pos: 本轮指令执行前无人机的当前位置
        """
        abs_waypoints = []
        for i, wp in enumerate(wps):
            print(f"{i} waypoint: \n-x: {wp['x']}\n-y: {wp['y']}\n-z: {wp['z']}\n{wp['description']}\n")

            abs_waypoints.append({
                "x": wp["x"] + cur_position.x_val,
                "y": wp["y"] + cur_position.y_val,
                "z": wp["z"] + cur_position.z_val,
                "description": wp["description"]
            })
        return abs_waypoints

    def step(
            self,
            instruction: str,
            image_rgb,
            cur_position,
            mode: str = "airsim",
            history: Optional[list[dict]] = None,
            extra_context: Optional[str] = None,            
            retries: Optional[int] = None,
    ) -> str:
        """
        完整的单步决策：instruct + img -> build_prompt → get_response → parse_response
        带重试机制。
        返回: 标准动作字符串
        """
        retries = retries or self.max_retries
        messages = self._build_prompt(instruction, image_rgb, history, extra_context, mode)

        for attempt in range(retries):
            try:
                raw = self._get_response(messages)
                action = self._parse_response(raw, cur_position)
                return action
            except Exception as e:
                print(f"[WARN] get_response attempt {attempt+1} failed: {e}")
                if attempt == retries - 1:
                    print("[ERROR] All retries failed, defaulting to 'forward'")
                    return "forward"