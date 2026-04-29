import os, time, logging
from openai import OpenAI
from .base_backend import VLMBackendBase

class CloudQwenBackend(VLMBackendBase):

    def __init__(self, cfg: dict):
        self.client = OpenAI(
            api_key=cfg.get("api_key") or os.environ["QWEN_API_KEY"],
            base_url=cfg.get("base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model_name    = cfg.get("model_name", "qwen-vl-max")
        self.max_tokens    = cfg.get("max_tokens", 1024)
        self.temperature   = cfg.get("temperature", 0.1)
        self.max_retries   = cfg.get("max_retries", 3)

    def query(self, messages: list[dict], **kwargs) -> str:
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                resp  = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                logging.info(f"API response in {time.time()-start:.2f}s")
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logging.warning(f"API attempt {attempt+1} failed: {e}")
        raise RuntimeError("CloudQwen: all retries failed")