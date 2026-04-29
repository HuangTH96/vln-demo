import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_backend import VLMBackendBase

class LocalQwenBackend(VLMBackendBase):

    def __init__(self, cfg: dict):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg["model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(cfg["model_path"])
        self.max_new_tokens = cfg.get("max_new_tokens", 256)

    def query(self, messages: list[dict], **kwargs) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()