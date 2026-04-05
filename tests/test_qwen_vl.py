# 该文件用来测试本地部署的qwen-vl模型是否可用
import torch
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("=== 环境检测 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n=== 加载模型 ===")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_path = os.path.expanduser("~/mLLM/qwen3-vl-4b")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,          # 你的本地路径
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
print("✅ 模型加载成功")

print("\n=== 简单推理测试 ===")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "你好，请用一句话介绍你自己"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
print(f"模型回复: {output[0]}")
print("\n✅ 全部检测通过！")