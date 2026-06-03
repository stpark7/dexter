"""Data transforms for Dexter.

This module provides data transformation utilities adapted from the workspace codebase.
Transforms operate on dictionary data structures and can be composed together.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, PreTrainedTokenizerBase

from dexter.data.normalize import NormStats, load
from dexter.models.action_tokenizer import GraspTokenizerQwen3
from dexter.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Type alias for data dictionaries
DataDict = dict[str, Any]

# ============================================================================
# ECoT Meta Prompt Pool for Contact Reasoning
# ============================================================================

# ECoT meta prompts (used when training_stage == 2)
# Full contact mode: joint names + positions
ECOT_META_PROMPTS_STAGE2_FULL = [
    # Natural & Direct (5 variants)
    "First predict which joints contact where on the object, then predict the grasp pose. Query: {task}",
    "Predict which parts of the hand touch where, then predict the grasp. Query: {task}",
    "Figure out which joints contact where first, then the grasp pose. Query: {task}",
    "Predict which hand joints touch the object and where, then the grasp configuration. Query: {task}",
    "Predict the contacting joints and their positions, then the grasp. Query: {task}",
    # Step-by-step reasoning (4 variants)
    "Think step by step: first predict which joints contact where, then predict the grasp pose. Query: {task}",
    "Let's break this down: identify which joints contact where, then determine the grasp. Query: {task}",
    "First identify which joints contact and their positions, then generate the grasp. Query: {task}",
    "Start by predicting which joints touch and where, then move on to the grasp pose. Query: {task}",
    # Reasoning-focused (3 variants)
    "Think about which hand joints will touch the object and where, then plan the grasp. Query: {task}",
    "Analyze which joints will contact where, then predict the grasp. Query: {task}",
    "Consider which joints will touch and their positions, then determine the grasp action. Query: {task}",
    # Action-oriented (3 variants)
    "Determine which hand joints contact where on the object before predicting the grasp action. Query: {task}",
    "Predict which joints will contact where, then predict the grasp pose. Query: {task}",
    "Identify the contacting joints and their positions, then predict the action. Query: {task}",
]

# Joint names only mode: controllable generation (no positions)
ECOT_META_PROMPTS_STAGE2_NAMES_ONLY = [
    # Natural & Direct (5 variants)
    "First predict which joints contact the object, then predict the grasp pose. Query: {task}",
    "Predict which parts of the hand touch the object, then predict the grasp. Query: {task}",
    "Figure out which joints contact the object first, then the grasp pose. Query: {task}",
    "Predict which hand joints touch the object, then the grasp configuration. Query: {task}",
    "Predict the contacting joints, then the grasp. Query: {task}",
    # Step-by-step reasoning (4 variants)
    "Think step by step: first predict which joints contact, then predict the grasp pose. Query: {task}",
    "Let's break this down: identify which joints contact the object, then determine the grasp. Query: {task}",
    "First identify which joints contact, then generate the grasp. Query: {task}",
    "Start by predicting which joints touch the object, then move on to the grasp pose. Query: {task}",
    # Reasoning-focused (3 variants)
    "Think about which hand joints will touch the object, then plan the grasp. Query: {task}",
    "Analyze which joints will contact the object, then predict the grasp. Query: {task}",
    "Consider which joints will touch the object, then determine the grasp action. Query: {task}",
    # Action-oriented (3 variants)
    "Determine which hand joints contact the object before predicting the grasp action. Query: {task}",
    "Predict which joints will contact the object, then predict the grasp pose. Query: {task}",
    "Identify the contacting joints, then predict the action. Query: {task}",
]


# ============================================================================
# Transform Builder
# ============================================================================
def build_preprocess_and_postprocess_transforms(
    model_cfg: DictConfig, data_cfg: DictConfig, split: str = "train", model=None
):
    """Build preprocess and postprocess transform pipeline from config using Hydra instantiate.

    Args:
        model_cfg: Model config
        data_cfg: Data config
        split: Dataset split (train/val/test)
        model: Optional model instance (required for Qwen3 models to resize embeddings)
    """
    assert data_cfg.norm_stats_path is not None, "Norm stats path is required"
    assert Path(data_cfg.norm_stats_path).exists(), "Norm stats path does not exist"
    norm_stats = load(data_cfg.norm_stats_path)

    assert model_cfg.tokenizer.model_id is not None, "Tokenizer model is required"
    processor = AutoProcessor.from_pretrained(model_cfg.tokenizer.model_id)
    if hasattr(processor, "tokenizer"):
        base_tokenizer = processor.tokenizer
    else:
        base_tokenizer = processor
    # base_tokenizer = AutoProcessor.from_pretrained(model_cfg.tokenizer.model_id).tokenizer

    # All Dexter models add special action/position/joint tokens and resize embeddings.
    if model is None or not hasattr(model, "setup_special_tokens"):
        raise ValueError(
            "build_preprocess_and_postprocess_transforms requires a model exposing "
            "setup_special_tokens()"
        )
    model.setup_special_tokens(
        base_tokenizer=base_tokenizer,
        n_action_bins=model_cfg.action.bins,
        n_position_bins=model_cfg.action.get("position_bins", 256),
    )

    action_tokenizer = GraspTokenizerQwen3(
        base_tokenizer,
        bins=model_cfg.action.bins,
        min_action=model_cfg.action.min_action,
        max_action=model_cfg.action.max_action,
        position_bins=model_cfg.action.get("position_bins", 256),
        min_position=model_cfg.action.get("min_position", -0.2),
        max_position=model_cfg.action.get("max_position", 0.2),
    )
    log.info("✓ Created GraspTokenizerQwen3 with special tokens")

    preprocess_transforms = build_transforms(
        data_cfg.transforms if split == "train" else data_cfg.eval_transforms,
        norm_stats,
        base_tokenizer,
        action_tokenizer,
    )
    postprocess_transforms = build_transforms(
        data_cfg.post_transforms, norm_stats, base_tokenizer, action_tokenizer
    )
    log.info(f"✓ Built preprocess transforms: {preprocess_transforms}")
    log.info(f"✓ Built postprocess transforms: {postprocess_transforms}")

    return preprocess_transforms, postprocess_transforms, base_tokenizer, action_tokenizer


def build_transforms(transform_configs, norm_stats, base_tokenizer, action_tokenizer):
    transform_list = []

    # Iterate through transforms in order (dict preserves order in Python 3.7+)
    for key, t_config in transform_configs.items():
        # Skip disabled transforms
        if not t_config.get("enabled", True):
            continue

        # Create a mutable copy for override
        cfg_copy = OmegaConf.create(OmegaConf.to_container(t_config, resolve=True))

        # Remove the 'enabled' flag (not a transform parameter)
        cfg_copy.pop("enabled", None)
        # Collect runtime parameters to inject after instantiation
        runtime_params = {}

        if "norm_stats" in cfg_copy and cfg_copy.norm_stats is None and norm_stats is not None:
            # Don't assign to config - OmegaConf doesn't support ndarray types
            runtime_params["norm_stats"] = norm_stats
            cfg_copy.pop("norm_stats", None)

        if (
            "base_tokenizer" in cfg_copy
            and cfg_copy.base_tokenizer is None
            and base_tokenizer is not None
        ):
            runtime_params["base_tokenizer"] = base_tokenizer
            cfg_copy.pop("base_tokenizer", None)

        if (
            "action_tokenizer" in cfg_copy
            and cfg_copy.action_tokenizer is None
            and action_tokenizer is not None
        ):
            runtime_params["action_tokenizer"] = action_tokenizer
            cfg_copy.pop("action_tokenizer", None)

        # Instantiate the transform using Hydra and inject runtime params
        transform = instantiate(cfg_copy, **runtime_params)
        transform_list.append(transform)

    return CompositeTransform(transform_list)


@dataclass(frozen=True)
class CompositeTransform:
    """A composite transform that applies a sequence of transforms in order.

    Example:
        transform = CompositeTransform([
            DexGYSInputs(normalize=True),
            RotateSceneZ(angle_range=(0, np.pi/2)),
            Normalize(norm_stats=stats)
        ])
        data = transform(data)
    """

    transforms: Sequence["Transform"]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __str__(self) -> str:
        transforms_str = "\n  ".join(str(t) for t in self.transforms)
        return f"CompositeTransform([\n  {transforms_str}\n])"


# ============================================================================
# Normalization Transforms
# ============================================================================


@dataclass(frozen=True)
class Normalize:
    """Normalize data using precomputed statistics.

    Supports both z-score normalization (mean/std) and quantile normalization (q01/q99).

    Args:
        norm_stats: Dictionary mapping field names to NormStats objects
        use_quantiles: If True, use quantile normalization instead of z-score
        strict: If True, raise error if norm_stats keys not found in data
    """

    norm_stats: dict[str, NormStats] | None
    use_quantiles: bool = False
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            for key, stats in self.norm_stats.items():
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(
                        f"Quantile stats must be provided if use_quantiles is True. Key '{key}' is missing q01 or q99."
                    )

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        for key, stats in self.norm_stats.items():
            if key not in data:
                if self.strict:
                    raise ValueError(f"Key '{key}' not found in data but required by norm_stats")
                continue

            x = data[key]
            if self.use_quantiles:
                data[f"{key}_norm"] = self._normalize_quantile(x, stats)
            else:
                data[f"{key}_norm"] = self._normalize(x, stats)

        return data

    def _normalize(self, x, stats: NormStats):
        """Z-score normalization."""
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        # Ensure numpy arrays have the same dtype as x
        mean = mean.astype(x.dtype)
        std = std.astype(x.dtype)
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        """Quantile normalization to [-1, 1] range."""
        assert stats.q01 is not None and stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        # Ensure numpy arrays have the same dtype as x
        q01 = q01.astype(x.dtype)
        q99 = q99.astype(x.dtype)
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def __str__(self) -> str:
        keys = list(self.norm_stats.keys()) if self.norm_stats else []
        return f"Normalize(keys={keys}, use_quantiles={self.use_quantiles}, strict={self.strict})"


@dataclass(frozen=True)
class Unnormalize:
    """Reverse normalization using precomputed statistics.

    Args:
        norm_stats: Dictionary mapping field names to NormStats objects
        use_quantiles: If True, reverse quantile normalization instead of z-score
    """

    norm_stats: dict[str, NormStats] | None
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            for key, stats in self.norm_stats.items():
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(
                        f"Quantile stats must be provided if use_quantiles is True. Key '{key}' is missing q01 or q99."
                    )

    def prepare_input(self, data: DataDict, stats) -> DataDict:
        actions = data["actions_norm"]
        if isinstance(actions, torch.Tensor):
            stats = {k: torch.from_numpy(v).to(actions.device) for k, v in stats.items()}
        return stats

    def __call__(self, data: DataDict) -> DataDict:
        assert self.norm_stats is not None, "Norm stats are required"

        for key, stats in self.norm_stats.items():
            stats = self.prepare_input(data, self.norm_stats[key])

            x = data[f"{key}_norm"]
            if self.use_quantiles:
                data[key] = self._unnormalize_quantile(x, stats)
            else:
                data[key] = self._unnormalize(x, stats)

        return data

    def _unnormalize(self, x, stats: dict):
        """Reverse z-score normalization."""
        mean = pad_to_dim(stats["mean"], x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats["std"], x.shape[-1], axis=-1, value=1.0)
        # Ensure numpy arrays have the same dtype as x
        mean = mean.astype(x.dtype) if isinstance(mean, np.ndarray) else mean.to(x.dtype)
        std = std.astype(x.dtype) if isinstance(std, np.ndarray) else std.to(x.dtype)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: dict):
        """Reverse quantile normalization from [-1, 1] range."""
        assert stats["q01"] is not None and stats["q99"] is not None
        q01, q99 = stats["q01"], stats["q99"]
        # Ensure numpy arrays have the same dtype as x
        q01 = q01.astype(x.dtype) if isinstance(q01, np.ndarray) else q01.to(x.dtype)
        q99 = q99.astype(x.dtype) if isinstance(q99, np.ndarray) else q99.to(x.dtype)
        if (dim := q01.shape[-1]) < x.shape[-1]:
            if isinstance(x, np.ndarray):
                return np.concatenate(
                    [
                        (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01,
                        x[..., dim:],
                    ],
                    axis=-1,
                )
            else:
                return torch.cat(
                    [
                        (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01,
                        x[..., dim:],
                    ],
                    dim=-1,
                )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def __str__(self) -> str:
        keys = list(self.norm_stats.keys()) if self.norm_stats else []
        return f"Unnormalize(keys={keys}, use_quantiles={self.use_quantiles})"


# ============================================================================
# Action Space Transforms
# ============================================================================


@dataclass(frozen=True)
class DetokenizeAction:
    action_tokenizer: GraspTokenizerQwen3

    def __call__(self, data: DataDict) -> DataDict:
        assert "tokenized_actions" in data, "Tokenized actions are required for detokenization"
        tokenized_actions = data["tokenized_actions"]
        actions = self.action_tokenizer.decode_token_ids_to_actions(tokenized_actions)
        data["actions_norm"] = actions
        return data

    def __str__(self) -> str:
        return f"DetokenizeAction(action_tokenizer={self.action_tokenizer})"


# ============================================================================
# Point Cloud Augmentation
# ============================================================================


@dataclass(frozen=True)
class RotateSceneZ:
    """Rotate point cloud and hand pose around z-axis for data augmentation.

    Applies consistent rotation to:
    - Point cloud data (if present)
    - Hand pose state (if present and rotate_state=True)
    - Hand pose actions (if present and rotate_actions=True)

    Rotation is performed around the point cloud's centroid in the x-y plane.

    Args:
        angle: Rotation angle in radians. If None, random rotation is applied.
        angle_range: Range for random rotation angles (min, max) in radians
        pointcloud_key: Key in data dict where point cloud is stored
        rotate_state: Whether to rotate the state (current hand pose)
        rotate_actions: Whether to rotate the actions (target hand pose sequence)
    """

    angle_range: tuple[float, float] = (0.0, 2 * np.pi)

    def __call__(self, data: DataDict) -> DataDict:
        assert "pointcloud" in data, "Point cloud is required for rotation"
        assert "actions" in data, "Actions are required for rotation"
        assert self.angle_range[0] <= self.angle_range[1], "Angle range must be non-negative"

        # Determine rotation angle
        rotation_angle = (
            self.angle_range[0]
            if self.angle_range[0] == self.angle_range[1]
            else np.random.uniform(self.angle_range[0], self.angle_range[1])
        )

        # Copy data to avoid modifying original
        data = data.copy()

        # Compute rotation center from point cloud (x-y centroid)
        center = np.array([0.0, 0.0, 0.0])
        pc = data["pointcloud"]
        center[:2] = np.mean(pc[..., :2], axis=-2)

        # Rotate point cloud if present
        pc[..., :2] -= center[:2]
        pc = rotate_point_cloud_z(pc, rotation_angle)
        pc[..., :2] += center[:2]

        data["pointcloud"] = pc

        # Rotate actions
        actions = data["actions"]
        # Adjust translation for rotation around centroid
        actions[..., :2] -= center[:2]
        actions = rotate_hand_pose_z(actions, rotation_angle)
        actions[..., :2] += center[:2]

        data["actions"] = actions

        if "contact" in data:
            contact = data["contact"]
            contact = rotate_contact_points(contact, rotation_angle, center)
            data["contact"] = contact

        return data

    def __str__(self) -> str:
        return f"RotateSceneZ(angle_range={self.angle_range})"


# ============================================================================
# DexGYS-Specific Transforms
# ============================================================================


@dataclass(frozen=True)
class NormalizePointCloud:
    """Normalize point cloud to unit sphere."""

    def __call__(self, data: DataDict) -> DataDict:
        pointcloud = data["pointcloud"].copy()
        pointcloud = pc_normalize(pointcloud)
        data["pointcloud"] = pointcloud
        return data

    def __str__(self) -> str:
        return "NormalizePointCloud()"


# ============================================================================
# Prompt & Padding Transforms
# ============================================================================


@dataclass(frozen=False)
class TokenizePromptQwen3:
    """Tokenize text prompts for Qwen3-VL based models with chat template support.

    This transform is specifically designed for Qwen3-VL models and includes:
    - Chat template formatting using apply_chat_template
    - Vision token injection for point cloud data
    - Proper handling of vision placeholders
    - Unified contact tokenization (encodes contact dict directly)
    - ECoT support with contact reasoning (training_stage 2)

    Contact data is automatically encoded from data["contact"] if present.
    Pre-computed data["contact_tokens"] is used if available (backwards compatibility).

    Args:
        action_tokenizer: GraspTokenizerQwen3 for action and contact encoding
        base_tokenizer: HuggingFace tokenizer (must support chat_template)
        max_length: Maximum token length
        training_stage: None (action-only, no ECoT) or 2 (full ECoT)
        contact_position_dropout: Probability of dropping contact positions (0.0 to 1.0).
                                  0.0 = always include positions, 1.0 = never include positions.
                                  Use 0.5 for balanced training with both modes.
    """

    action_tokenizer: GraspTokenizerQwen3
    base_tokenizer: PreTrainedTokenizerBase
    max_length: int = 128
    vision_token: str = "<|vision_pad|>"
    is_training: bool = True
    force_cot: bool = False
    training_stage: int | None = None
    contact_position_dropout: float = 0.0
    postfix_contact_string: str | None = None
    steer_link_num: int | None = None

    def __call__(self, data: DataDict) -> DataDict:
        # Get prompt from data
        prompt = data.get("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = str(prompt)

        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")

        # Check if tokenizer has a chat template (required for this transform)
        if (
            not hasattr(self.base_tokenizer, "chat_template")
            or self.base_tokenizer.chat_template is None
        ):
            raise ValueError(
                "TokenizePromptQwen3 requires a tokenizer with chat_template support. "
                "Use a tokenizer that provides a chat template."
            )

        has_vision_tokens = self.vision_token in self.base_tokenizer.added_tokens_encoder
        assert has_vision_tokens, "Tokenizer does not have vision tokens"

        vision_placeholder = "<|vision_start|>" + self.vision_token + "<|vision_end|>"

        include_positions = True
        if not self.force_cot or (self.is_training and self.contact_position_dropout > 0.0):
            include_positions = random.random() > self.contact_position_dropout

        contact_tokens = data.get("contact_tokens", None)
        if contact_tokens is None and "contact" in data:
            contact_dict = data.get("contact", {})
            contact_tokens = (
                self.action_tokenizer.encode_contact(
                    contact_dict,
                    include_positions=include_positions,
                    steer_link_num=self.steer_link_num,
                )
                if contact_dict
                else []
            )
            data["contact_tokens"] = contact_tokens

        # Use ECoT meta prompts when ECoT is enabled (training_stage == 2)
        use_ecot = self.training_stage == 2 and contact_tokens
        if use_ecot:
            # Pick the prompt pool based on whether we include contact positions
            prompt_pool = (
                ECOT_META_PROMPTS_STAGE2_FULL
                if include_positions
                else ECOT_META_PROMPTS_STAGE2_NAMES_ONLY
            )
            meta_prompt = random.choice(prompt_pool) if self.is_training else prompt_pool[0]
            instruction_prompt = meta_prompt.format(task=cleaned_text)
        else:
            instruction_prompt = f"Given the grasp description, generate the hand pose to achieve the grasp. Description: {cleaned_text}"

        user_content = (
            vision_placeholder + instruction_prompt if has_vision_tokens else instruction_prompt
        )
        messages = [{"role": "user", "content": user_content}]

        contact_strings = None
        if "contact_tokens" in data:
            contact_token_ids = data["contact_tokens"]
            contact_strings = self.base_tokenizer.decode(contact_token_ids)

        if self.is_training:
            # Build assistant content
            if self.training_stage == 2:
                # Full ECoT: contact reasoning + action
                action_strings = self.action_tokenizer.action_to_token_strings(data["actions_norm"])
                assistant_content = contact_strings + "".join(action_strings)
            else:
                # No ECoT: action only
                action_strings = self.action_tokenizer.action_to_token_strings(data["actions_norm"])
                assistant_content = "".join(action_strings)

            messages.append({"role": "assistant", "content": assistant_content})
        else:
            if self.steer_link_num is not None:
                messages.append({"role": "assistant", "content": contact_strings})

        formatted_text = self.base_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # add generation prompt for evaluation, not for training
            add_generation_prompt=not self.is_training and self.steer_link_num is None,
            continue_final_message=self.steer_link_num is not None,
        )
        tokenized = self.base_tokenizer(
            formatted_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
        )

        tokenized_prompt = tokenized["input_ids"][0]
        tokenized_prompt_mask = tokenized["attention_mask"][0]

        labels = tokenized_prompt.copy()
        labels[tokenized_prompt_mask == 0] = -100

        if self.is_training:
            user_only_text = self.base_tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            user_tokens = self.base_tokenizer.encode(user_only_text, add_special_tokens=False)
            labels[: len(user_tokens)] = -100

        # Store vision token information for model to use
        result = {
            **data,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            "labels": labels,
        }

        # Store vision token ID for model to identify and replace
        if self.vision_token in self.base_tokenizer.added_tokens_encoder:
            result["vision_token_id"] = self.base_tokenizer.added_tokens_encoder[self.vision_token]

        return result

    def __str__(self) -> str:
        return ", ".join(
            [
                f"TokenizePromptQwen3(action_tokenizer={self.action_tokenizer}",
                f"base_tokenizer={self.base_tokenizer.__class__.__name__}",
                f"max_length={self.max_length}",
                f"vision_token={self.vision_token}",
                f"training_stage={self.training_stage}",
                f"contact_position_dropout={self.contact_position_dropout})",
            ]
        )


# ============================================================================
# Data Type Transforms
# ============================================================================


@dataclass(frozen=True)
class ToTensor:
    """Convert all NumPy arrays in data dict to PyTorch tensors.

    Iterates over all fields and converts NumPy arrays to tensors.
    Fields that are already tensors or other types are left unchanged.
    """

    def __call__(self, data: DataDict) -> DataDict:
        data = data.copy()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
        return data

    def __str__(self) -> str:
        return "ToTensor()"


# ============================================================================
# Utility Functions
# ============================================================================


def pad_to_dim(
    x: np.ndarray | torch.Tensor, target_dim: int, axis: int = -1, value: float = 0.0
) -> np.ndarray | torch.Tensor:
    """Pad an array to the target dimension along the specified axis.

    Args:
        x: Input array
        target_dim: Target dimension size
        axis: Axis to pad along
        value: Padding value

    Returns:
        Padded array
    """
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        if isinstance(x, np.ndarray):
            return np.pad(x, pad_width, constant_values=value)
        elif isinstance(x, torch.Tensor):
            return torch.nn.functional.pad(x, pad_width, value=value)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")
    return x


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere.

    Args:
        pc: Point cloud array [..., N, C] where C >= 3

    Returns:
        Normalized point cloud with same shape
    """
    centroid = np.mean(pc[..., :3], axis=-2, keepdims=True)
    pc_centered = pc.copy()
    pc_centered[..., :3] -= centroid
    max_dist = np.max(np.sqrt(np.sum(pc_centered[..., :3] ** 2, axis=-1)))
    pc_centered[..., :3] /= max_dist + 1e-8
    return pc_centered


def rotate_point_cloud_z(pc: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point cloud around z-axis.

    Args:
        pc: Point cloud array [..., N, C] where C >= 3
        angle: Rotation angle in radians

    Returns:
        Rotated point cloud with same shape
    """
    original_dtype = pc.dtype
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Rotation matrix for z-axis
    rotation_matrix = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]],
        dtype=original_dtype,
    )

    # Extract xyz coordinates
    xyz = pc[..., :3]

    # Apply rotation
    rotated_xyz = xyz @ rotation_matrix.T

    # Keep additional features unchanged
    if pc.shape[-1] > 3:
        return np.concatenate([rotated_xyz, pc[..., 3:]], axis=-1)
    return rotated_xyz


def rotate_contact_points(
    contact: dict[str, np.ndarray], angle: float, center: np.ndarray
) -> dict[str, np.ndarray]:
    """Rotate contact points around z-axis.

    Args:
        contact: Dictionary mapping joint names to contact points
        angle: Rotation angle in radians

    Returns:
        Rotated contact points with same shape and dtype
    """
    new_contact = {}
    for joint_name, points in contact.items():
        new_points = points.copy()
        new_points[..., :2] = new_points[..., :2] - center[:2]
        new_points = rotate_point_cloud_z(new_points, angle)
        new_points[..., :2] = new_points[..., :2] + center[:2]
        new_contact[joint_name] = new_points
    return new_contact


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    Args:
        axis_angle: [..., 3] array of axis-angle rotations

    Returns:
        rotation_matrix: [..., 3, 3] array of rotation matrices
    """
    original_dtype = axis_angle.dtype

    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)

    # Handle zero angle case
    small_angle = angle < 1e-8
    default_axis = np.array([1.0, 0.0, 0.0], dtype=original_dtype)
    axis = np.where(small_angle, default_axis, axis_angle / (angle + 1e-8))

    angle = angle.squeeze(-1)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle

    # Extract axis components
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    # Build rotation matrix using Rodrigues' formula
    shape = axis_angle.shape[:-1] + (3, 3)
    rotation_matrix = np.zeros(shape, dtype=original_dtype)

    # Diagonal elements
    rotation_matrix[..., 0, 0] = cos_angle + x * x * one_minus_cos
    rotation_matrix[..., 1, 1] = cos_angle + y * y * one_minus_cos
    rotation_matrix[..., 2, 2] = cos_angle + z * z * one_minus_cos

    # Off-diagonal elements
    rotation_matrix[..., 0, 1] = x * y * one_minus_cos - z * sin_angle
    rotation_matrix[..., 0, 2] = x * z * one_minus_cos + y * sin_angle
    rotation_matrix[..., 1, 0] = y * x * one_minus_cos + z * sin_angle
    rotation_matrix[..., 1, 2] = y * z * one_minus_cos - x * sin_angle
    rotation_matrix[..., 2, 0] = z * x * one_minus_cos - y * sin_angle
    rotation_matrix[..., 2, 1] = z * y * one_minus_cos + x * sin_angle

    # Handle zero angle case - should be identity matrix
    identity = np.eye(3, dtype=original_dtype)
    rotation_matrix = np.where(small_angle[..., None], identity, rotation_matrix)

    return rotation_matrix


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation.

    Args:
        matrix: [..., 3, 3] array of rotation matrices

    Returns:
        axis_angle: [..., 3] array of axis-angle rotations
    """
    original_dtype = matrix.dtype

    # Compute the rotation angle
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    # Handle small angles (close to identity)
    small_angle = angle < 1e-6

    # Compute the rotation axis
    axis = np.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )

    # Normalize the axis
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / (axis_norm + 1e-8)

    # Multiply axis by angle
    axis_angle = axis * angle[..., None]

    # Handle small angle case - return zero
    axis_angle = np.where(small_angle[..., None], 0.0, axis_angle)

    return axis_angle.astype(original_dtype)


def rotate_hand_pose_z(hand_pose: np.ndarray, angle: float) -> np.ndarray:
    """Rotate hand pose around z-axis.

    Args:
        hand_pose: [..., 28] array where:
            [0:3] - global translation (x, y, z)
            [3:6] - global rotation as axis-angle
            [6:] - joint angles
        angle: Rotation angle in radians

    Returns:
        Rotated hand pose with same shape and dtype
    """
    original_dtype = hand_pose.dtype

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Create z-rotation matrix
    R_z = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]],
        dtype=original_dtype,
    )

    # Rotate translation (x, y) coordinates
    translation = hand_pose[..., :3].copy()
    translation[..., :2] = translation[..., :2] @ R_z[:2, :2].T

    # Compose rotation: R' = R_z @ R
    current_rotation = hand_pose[..., 3:6]
    R_current = axis_angle_to_matrix(current_rotation)
    R_new = R_z @ R_current
    new_rotation = matrix_to_axis_angle(R_new)

    # Keep joint angles unchanged
    joints = hand_pose[..., 6:]

    # Concatenate all components
    result = np.concatenate([translation, new_rotation, joints], axis=-1)
    return result.astype(original_dtype)


# Type alias for transforms
Transform = (
    CompositeTransform
    | Normalize
    | Unnormalize
    | RotateSceneZ
    | NormalizePointCloud
    | DetokenizeAction
    | TokenizePromptQwen3
    | ToTensor
)
