from typing import Optional

from transformers import PretrainedConfig


class EncoderConfig(PretrainedConfig):
    """Configuration for the point cloud encoder (Uni3D or PartField)."""

    def __init__(
        self,
        type: str = "uni3d",
        variant: str = "base",
        normalize_point_cloud: bool = True,
        downsample_patch_embeddings: bool = False,
        freeze: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.variant = variant
        self.normalize_point_cloud = normalize_point_cloud
        self.downsample_patch_embeddings = downsample_patch_embeddings
        self.freeze = freeze


class VLMConfig(PretrainedConfig):
    """Configuration for the vision-language model."""

    def __init__(
        self,
        model_id: str = "google/paligemma-3b-pt-224",
        variant: str = "gemma_2b",
        freeze: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.variant = variant
        self.freeze = freeze


class TokenizerConfig(PretrainedConfig):
    """Configuration for the tokenizer."""

    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id


class ShadowHandConfig(PretrainedConfig):
    """Configuration for the shadow hand model."""

    def __init__(self, base_dir: str = "./assets/shadowhand", **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir


class DexterConfig(PretrainedConfig):
    """Shared configuration for autoregressive Dexter model variants.

    Backbone-specific subclasses only need to set `model_type`.
    """

    model_type = "dexter"

    def __init__(
        self,
        # ===== Nested Configs =====
        encoder: dict = None,
        vlm: dict = None,
        tokenizer: dict = None,
        # ===== Action Space =====
        action_dim: int = 32,
        action_horizon: int = 1,
        use_state_input: bool = False,
        # ===== Autoregressive Settings =====
        num_action_bins: int = 256,
        # ===== Hand Model =====
        shadowhand: dict = None,
        # ===== Training =====
        attention_type: str = "sdpa",
        gradient_checkpointing: bool = False,
        label_smoothing: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Nested configs
        self.encoder = EncoderConfig(**(encoder or {}))
        self.vlm = VLMConfig(**(vlm or {}))
        self.tokenizer = TokenizerConfig(**(tokenizer or {}))
        self.shadowhand = ShadowHandConfig(**(shadowhand or {}))

        # Action space
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_state_input = use_state_input

        # Autoregressive settings
        self.num_action_bins = num_action_bins

        # Training
        self.attention_type = attention_type
        self.gradient_checkpointing = gradient_checkpointing
        self.label_smoothing = label_smoothing


class DexterQwenConfig(DexterConfig):
    """Configuration for Dexter with a Qwen3-VL / Qwen2.5 backbone."""

    model_type = "dexter_qwen3"


class DexterSmolLM2Config(DexterConfig):
    """Configuration for Dexter with a SmolLM2 (Llama architecture) backbone."""

    model_type = "dexter_smollm2"
