from .base import Model
from .claude import ClaudeHighModel, ClaudeLowModel, ClaudeMediumModel
from .deepseek_r1 import DeepSeekR1Model
from .deepseek_v31 import DeepSeekV31Model
from .gemini import GeminiModel
from .glm45 import GLM45Model
from .gpt_oss import GPTOssModel
from .gpt_think import GPTThinkHighModel, GPTThinkLowModel, GPTThinkMediumModel
from .kimi_k2 import KimiK2Model
from .minimax_m1 import MinimaxM1Model
from .qwen3 import Qwen3Model
from .qwen3_think import Qwen3ThinkModel

MODELS: dict[str, type[Model]] = {
    model_cls.name(): model_cls
    for model_cls in (
        GPTThinkLowModel,
        GPTThinkMediumModel,
        GPTThinkHighModel,
        GPTOssModel,
        GeminiModel,
        ClaudeLowModel,
        ClaudeMediumModel,
        ClaudeHighModel,
        DeepSeekR1Model,
        DeepSeekV31Model,
        Qwen3ThinkModel,
        Qwen3Model,
        MinimaxM1Model,
        KimiK2Model,
        GLM45Model,
    )
}

__all__ = ["Model", "MODELS"]
