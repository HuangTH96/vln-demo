from .airsim_prompts import SYSTEM_PROMPT_SIM
from .tello_prompts import SYSTEM_PROMPT_TELLO


PROMPT_REGISTRY = {
    "airsim": SYSTEM_PROMPT_SIM,
    "tello": SYSTEM_PROMPT_TELLO
}

def get_prompt(
        platform: str             
) -> str:
    
    if platform not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown platform: {platform}. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )
    
    return PROMPT_REGISTRY[platform]