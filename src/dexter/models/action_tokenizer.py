"""
action_tokenizer.py

Action tokenizer with special tokens for action bins, position bins, and joint names.
This module provides a centralized function to add special tokens and resize model embeddings.
"""

import re
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from dexter.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Common contact joint names for Shadow Hand
CONTACT_JOINT_NAMES = [
    "rh_palm",
    "rh_ffdistal",
    "rh_ffmiddle",
    "rh_ffproximal",
    "rh_ffknuckle",
    "rh_mfdistal",
    "rh_mfmiddle",
    "rh_mfproximal",
    "rh_mfknuckle",
    "rh_rfdistal",
    "rh_rfmiddle",
    "rh_rfproximal",
    "rh_rfknuckle",
    "rh_lfdistal",
    "rh_lfmiddle",
    "rh_lfproximal",
    "rh_lfknuckle",
    "rh_thdistal",
    "rh_thhub",
    "rh_thmiddle",
    "rh_thproximal",
]


def setup_special_tokens_and_resize_embeddings(
    tokenizer: PreTrainedTokenizerBase,
    model: nn.Module,
    n_action_bins: int = 256,
    n_position_bins: int = 256,
) -> int:
    """
    Add special tokens to tokenizer and resize model embeddings.

    This function is idempotent - it's safe to call multiple times.
    If tokens are already present (e.g., when loading a pretrained model),
    it will skip adding them and resizing embeddings.

    This is the SINGLE place where we:
    1. Add special tokens for action bins, position bins, and joint names
    2. Add vision tokens (vision_pad, vision_start, vision_end) if not present
    3. Resize the model's embedding layer to accommodate new tokens
    4. Initialize new embeddings appropriately

    Args:
        tokenizer: The base tokenizer (e.g., Qwen3 tokenizer or SmolLM2 tokenizer)
        model: The model whose embeddings need to be resized
        n_action_bins: Number of action bins
        n_position_bins: Number of position bins

    Returns:
        Number of tokens added
    """
    original_vocab_size = len(tokenizer)
    log.info(f"Original vocab size: {original_vocab_size}")

    # Check if special tokens are already present (idempotent check)
    test_token = "<action_bin_0>"
    if test_token in tokenizer.get_vocab():
        log.warning(
            f"Special tokens already present in tokenizer (vocab size: {original_vocab_size})"
        )
        log.warning("Skipping token addition and embedding resize")
        return 0

    special_tokens = []

    # Add vision tokens if not already present (needed for SmolLM2 and other LLMs without vision)
    # These are required by TokenizePromptQwen3 for multimodal input handling
    vision_tokens = ["<|vision_pad|>", "<|vision_start|>", "<|vision_end|>"]
    for vt in vision_tokens:
        if vt not in tokenizer.get_vocab():
            special_tokens.append(vt)
            log.info(f"Adding vision token: {vt}")

    # Add action bin tokens: <action_bin_0>, <action_bin_1>, ..., <action_bin_N-1>
    for i in range(n_action_bins):
        special_tokens.append(f"<action_bin_{i}>")

    special_tokens.append("<|action_start|>")
    special_tokens.append("<|action_end|>")

    # Add position bin tokens: <pos_bin_0>, <pos_bin_1>, ..., <pos_bin_M-1>
    for i in range(n_position_bins):
        special_tokens.append(f"<pos_bin_{i}>")

    special_tokens.append("<|pos_start|>")
    special_tokens.append("<|pos_end|>")

    # Add joint name tokens: <rh_palm>, <rh_ffdistal>, etc.
    for joint_name in CONTACT_JOINT_NAMES:
        special_tokens.append(f"<{joint_name}>")

    special_tokens.append("<|joint_start|>")
    special_tokens.append("<|joint_end|>")

    # Add all special tokens to the tokenizer
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    log.info(f"Added {num_added} special tokens to tokenizer")
    log.info(f"New vocab size: {len(tokenizer)}")

    # Resize model embeddings to match new tokenizer vocab size
    # This will add new embedding rows initialized with small random values
    model.resize_token_embeddings(len(tokenizer))
    log.info(f"Resized model embeddings to {len(tokenizer)}")

    # Get the embedding layer and initialize new embeddings
    # For Qwen3-VL, the embedding layer is at model.language_model.embed_tokens
    if hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
        embed_tokens = model.language_model.embed_tokens
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
    else:
        # Fallback: try to find embed_tokens
        embed_tokens = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and "embed_tokens" in name:
                embed_tokens = module
                break

    if embed_tokens is not None:
        # Initialize new embeddings with small random values
        # New embeddings are at indices [original_vocab_size, original_vocab_size + num_added)
        with torch.no_grad():
            new_embeddings = embed_tokens.weight[original_vocab_size:, :]
            # Initialize with small random values (mean=0, std=0.02)
            nn.init.normal_(new_embeddings, mean=0.0, std=0.02)
        log.info(f"Initialized {num_added} new embeddings")
    else:
        log.warning("Warning: Could not find embedding layer to initialize new tokens")

    return num_added


class GraspTokenizerQwen3:
    """
    Action tokenizer for Qwen3-based models.

    Uses dedicated special tokens (added to the tokenizer) rather than reusing the
    least-used tokens of the vocabulary, for:
    - Action bins: <action_bin_0>, <action_bin_1>, ..., <action_bin_N>
    - Position bins: <pos_bin_0>, <pos_bin_1>, ..., <pos_bin_M>
    - Joint names: <rh_palm>, <rh_ffdistal>, ..., etc.

    IMPORTANT: Before creating this tokenizer, you must call
    setup_special_tokens_and_resize_embeddings() to add the tokens and resize embeddings.
    """

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1,
        max_action: float = 1,
        position_bins: int = 256,
        min_position: float = -0.4,
        max_position: float = 0.4,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension using special tokens.

        NOTE: This assumes setup_special_tokens_and_resize_embeddings() has already been called!

        :param base_tokenizer: Base LLM/VLM tokenizer (with special tokens already added).
        :param bins: Number of bins for each continuous action value; uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        :param position_bins: Number of bins for 3D position values (for contact points).
        :param min_position: Minimum position value for 3D contact points.
        :param max_position: Maximum position value for 3D contact points.
        """
        self.base_tokenizer = base_tokenizer

        # Only set add_bos_token if the tokenizer has a BOS token
        # Qwen3-VL has no BOS token (null), so we skip this for Qwen
        if hasattr(base_tokenizer, "bos_token") and base_tokenizer.bos_token is not None:
            self.base_tokenizer.add_bos_token = True

        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        self.n_position_bins = position_bins
        self.min_position = min_position
        self.max_position = max_position

        # Create Uniform Bins + Compute Bin Centers for actions
        self.bins = np.linspace(min_action, max_action, self.n_bins, dtype=np.float32)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Create Uniform Bins + Compute Bin Centers for positions
        self.position_bins_array = np.linspace(
            min_position, max_position, self.n_position_bins, dtype=np.float32
        )
        self.position_bin_centers = (
            self.position_bins_array[:-1] + self.position_bins_array[1:]
        ) / 2.0

        # Setup token ID ranges (tokens should already be added to tokenizer)
        self._setup_token_id_ranges()

    def _setup_token_id_ranges(self) -> None:
        """
        Setup token ID ranges for action bins, position bins, and joint names.

        This looks up the token IDs from the tokenizer's vocabulary.
        """
        # Get token IDs for action bins
        action_token_ids = []
        for i in range(self.n_bins):
            token_id = self.base_tokenizer.convert_tokens_to_ids(f"<action_bin_{i}>")
            action_token_ids.append(token_id)

        self.action_token_begin_idx = min(action_token_ids)
        self.action_token_end_idx = max(action_token_ids) + 1

        # Get token IDs for position bins
        position_token_ids = []
        for i in range(self.n_position_bins):
            token_id = self.base_tokenizer.convert_tokens_to_ids(f"<pos_bin_{i}>")
            position_token_ids.append(token_id)

        self.position_token_begin_idx = min(position_token_ids)
        self.position_token_end_idx = max(position_token_ids) + 1

        # Get token IDs for joint names
        self.joint_name_to_token_id = {}
        for joint_name in CONTACT_JOINT_NAMES:
            token_id = self.base_tokenizer.convert_tokens_to_ids(f"<{joint_name}>")
            self.joint_name_to_token_id[joint_name] = token_id

        self.joint_token_begin_idx = min(self.joint_name_to_token_id.values())
        self.joint_token_end_idx = max(self.joint_name_to_token_id.values()) + 1

        # Create reverse mapping from token ID to joint name
        self.token_id_to_joint_name = {
            token_id: joint_name for joint_name, token_id in self.joint_name_to_token_id.items()
        }

        # Store the action token ID list for direct indexing
        self.action_token_ids = np.array(action_token_ids, dtype=np.int64)
        self.position_token_ids = np.array(position_token_ids, dtype=np.int64)

        # Create set for O(1) lookup of valid action bin tokens
        self.action_token_ids_set = set(action_token_ids)

        # Cache long tensors of the token-id groups for the is_*_token masks, which
        # run every training step in the loss. `.to(device)` on a matching tensor is
        # a no-op, so this avoids rebuilding the tensors on each call.
        self._action_token_ids_tensor = torch.tensor(action_token_ids, dtype=torch.long)
        self._position_token_ids_tensor = torch.from_numpy(self.position_token_ids).long()
        self._joint_token_ids_tensor = torch.tensor(
            list(self.joint_name_to_token_id.values()), dtype=torch.long
        )

        # Get special marker token IDs for ECoT structured generation
        self.action_start_token_id = self.base_tokenizer.convert_tokens_to_ids("<|action_start|>")
        self.action_end_token_id = self.base_tokenizer.convert_tokens_to_ids("<|action_end|>")
        self.joint_start_token_id = self.base_tokenizer.convert_tokens_to_ids("<|joint_start|>")
        self.joint_end_token_id = self.base_tokenizer.convert_tokens_to_ids("<|joint_end|>")

    def is_action_token(self, token_ids):
        """
        Check which tokens are action bin tokens (regardless of block context).

        Args:
            token_ids: Token IDs tensor of shape [B, L] (PyTorch tensor)

        Returns:
            Boolean mask of shape [B, L] where True indicates action bin tokens
        """
        action_token_ids_tensor = self._action_token_ids_tensor.to(token_ids.device)
        return (token_ids.unsqueeze(-1) == action_token_ids_tensor).any(dim=-1)

    def is_joint_token(self, token_ids):
        """
        Check which tokens are joint name tokens (regardless of block context).

        Args:
            token_ids: Token IDs tensor of shape [B, L] (PyTorch tensor)

        Returns:
            Boolean mask of shape [B, L] where True indicates joint name tokens
        """
        joint_token_ids_tensor = self._joint_token_ids_tensor.to(token_ids.device)
        return (token_ids.unsqueeze(-1) == joint_token_ids_tensor).any(dim=-1)

    def is_position_token(self, token_ids):
        """
        Check which tokens are position bin tokens (regardless of block context).

        Args:
            token_ids: Token IDs tensor of shape [B, L] (PyTorch tensor)

        Returns:
            Boolean mask of shape [B, L] where True indicates position bin tokens
        """
        position_token_ids_tensor = self._position_token_ids_tensor.to(token_ids.device)
        return (token_ids.unsqueeze(-1) == position_token_ids_tensor).any(dim=-1)

    def __str__(self) -> str:
        msg = [
            f"GraspTokenizerQwen3(base_tokenizer={self.base_tokenizer.__class__.__name__})",
            f"Vocab size: {len(self.base_tokenizer)}",
            f"Action bins: {self.n_bins}",
            f"  - min_action: {self.min_action}, max_action: {self.max_action}",
            f"  - token range: [{self.action_token_begin_idx}, {self.action_token_end_idx})",
            f"Position bins: {self.n_position_bins}",
            f"  - min_position: {self.min_position}, max_position: {self.max_position}",
            f"  - token range: [{self.position_token_begin_idx}, {self.position_token_end_idx})",
            f"Joint names: {len(CONTACT_JOINT_NAMES)}",
            f"  - token range: [{self.joint_token_begin_idx}, {self.joint_token_end_idx})",
        ]
        return "\n".join(msg)

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        Clip & bin actions to action bin tokens.

        Args:
            action: Single action [D] or batch of actions [B, D]

        Returns:
            String or list of strings representing the tokenized actions
        """
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)  # ranges from [1, n_bins]

        # Clip to valid range [1, n_bins]
        discretized_action = np.clip(discretized_action, 1, self.n_bins)

        # Map to token IDs using the lookup array
        # Subtract 1 because digitize returns 1-indexed bins
        token_ids = self.action_token_ids[discretized_action - 1]

        # Handle single element vs. batch
        if len(token_ids.shape) == 1:
            # Single action
            decode = self.base_tokenizer.decode(token_ids.tolist())
        else:
            # Batch of actions
            decode = self.base_tokenizer.batch_decode(token_ids.tolist())

        return decode

    def action_to_token_strings(self, action: np.ndarray):
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)  # ranges from [1, n_bins]
        discretized_action = discretized_action - 1  # ranges from [0, n_bins-1]
        token_strings = (
            ["<|action_start|>"]
            + [f"<action_bin_{i}>" for i in discretized_action]
            + ["<|action_end|>"]
        )
        return token_strings

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        Extracts bin index `x` from token names like `<action_bin_x>` using regex.
        Handles non-action tokens gracefully by defaulting to bin 0.

        Args:
            action_token_ids: Token IDs of shape [D] or [B, D]

        Returns:
            Continuous actions of shape [D] or [B, D]
        """
        shape = action_token_ids.shape
        flat_ids = action_token_ids.flatten()

        # Convert token IDs to token strings
        token_strings = [self.base_tokenizer.convert_ids_to_tokens(int(tid)) for tid in flat_ids]

        # Regex pattern to match <action_bin_123> and extract 123
        pattern = re.compile(r"<action_bin_(\d+)>")

        # Extract bin indices from token names
        bin_indices = []
        for token_string in token_strings:
            match = pattern.match(token_string)
            if match:
                # Extract the number from the matched group
                bin_idx = int(match.group(1))
                bin_indices.append(bin_idx)
            else:
                # Non-action token - default to bin 0
                bin_indices.append(0)

        bin_indices = np.array(bin_indices).reshape(shape)

        # Clip to valid range [0, n_bins-1]
        bin_indices = np.clip(bin_indices, 0, self.bin_centers.shape[0] - 1)

        # Map bin indices to continuous action values using bin centers
        return self.bin_centers[bin_indices]

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        return len(self.base_tokenizer)

    def encode_contact(
        self, contact_dict: dict, include_positions: bool = True, steer_link_num: int | None = None
    ) -> list:
        """
        Encode contact dict to token IDs for ECoT reasoning or controllable generation.

        Args:
            contact_dict: {'rh_ffmiddle': [[x,y,z]], 'rh_ffdistal': [[x,y,z]], ...}
            include_positions: If True, include position bins after each joint name token.
                              If False, only include joint name tokens (for controllable models).

        Returns:
            If include_positions=True:
                [joint_start_id, joint1_id, pos_x, pos_y, pos_z, ..., joint_end_id]
            If include_positions=False:
                [joint_start_id, joint1_id, joint2_id, ..., joint_end_id]
        """
        if not contact_dict:
            return [self.joint_start_token_id, self.joint_end_token_id]

        token_ids = [self.joint_start_token_id]

        N = steer_link_num if steer_link_num is not None else len(contact_dict)
        for joint_name in sorted(contact_dict.keys())[:N]:
            if joint_name not in self.joint_name_to_token_id:
                continue

            if include_positions:
                # Full contact: joint names + positions (for ECoT reasoning)
                positions = np.atleast_2d(contact_dict[joint_name])  # Ensure 2D: [[x,y,z], ...]

                for pos in positions:
                    pos = np.clip(pos[:3], self.min_position, self.max_position)
                    bins = np.digitize(pos, self.position_bins_array).clip(1, self.n_position_bins)
                    token_ids.append(self.joint_name_to_token_id[joint_name])
                    token_ids.extend(self.position_token_ids[bins - 1].tolist())
            else:
                # Joint names only (for controllable models)
                # Only add the joint name token once, regardless of number of contact points
                token_ids.append(self.joint_name_to_token_id[joint_name])

        if steer_link_num is None:
            token_ids.append(self.joint_end_token_id)
        return token_ids

    def parse_ecot_output(self, generated_ids: Union[list, str], action_dim: int) -> dict:
        """
        Parse full ECoT generation into contact reasoning and action.

        Args:
            generated_ids: Either list of token IDs or decoded string

        Returns:
            {
                'contact_reasoning': {'rh_ffmiddle': [[x,y,z]], ...},
                'predicted_action': [28-dim numpy array],
            }
        """
        # Convert to string if token IDs provided
        if isinstance(generated_ids, list):
            generated_str = self.base_tokenizer.decode(generated_ids)
        else:
            generated_str = generated_ids

        result = {
            "contact_reasoning": {},
            "predicted_action": None,
        }

        # Extract contact section between <|joint_start|> and <|joint_end|>
        joint_start_idx = generated_str.find("<|joint_start|>")
        joint_end_idx = generated_str.find("<|joint_end|>")

        # if joint_start_idx >= 0 and joint_end_idx > joint_start_idx:
        contact_str = generated_str[joint_start_idx + len("<|joint_start|>") : joint_end_idx]
        result["contact_reasoning"] = self._parse_contact_string(contact_str)

        # Extract action section between <|action_start|> and <|action_end|>
        action_start_idx = generated_str.find("<|action_start|>")
        action_end_idx = generated_str.find("<|action_end|>")

        # if action_start_idx >= 0 and action_end_idx > action_start_idx:
        action_str = generated_str[action_start_idx + len("<|action_start|>") : action_end_idx]
        result["predicted_action"] = self._parse_action_string(action_str, action_dim)

        return result

    def _parse_contact_string(self, contact_str: str) -> dict:
        """
        Parse contact string to extract joint positions.

        Args:
            contact_str: String like "<rh_palm><pos_bin_10><pos_bin_20><pos_bin_30>..."

        Returns:
            {'rh_ffmiddle': [[x,y,z]], ...}
        """
        contact_dict = {}

        # Find all joint patterns: <joint_name> followed by 3 <pos_bin_X> tokens
        for joint_name in CONTACT_JOINT_NAMES:
            # Find all occurrences of this joint
            pattern = rf"<{joint_name}><pos_bin_(\d+)><pos_bin_(\d+)><pos_bin_(\d+)>"
            matches = re.finditer(pattern, contact_str)

            for match in matches:
                # Extract position bin indices
                bin_x, bin_y, bin_z = int(match.group(1)), int(match.group(2)), int(match.group(3))

                # Clip to valid range
                bin_x = np.clip(bin_x, 0, len(self.position_bin_centers) - 1)
                bin_y = np.clip(bin_y, 0, len(self.position_bin_centers) - 1)
                bin_z = np.clip(bin_z, 0, len(self.position_bin_centers) - 1)

                # Decode to continuous position
                position = [
                    self.position_bin_centers[bin_x],
                    self.position_bin_centers[bin_y],
                    self.position_bin_centers[bin_z],
                ]

                contact_dict.setdefault(joint_name, []).append(position)

        return contact_dict

    def _parse_action_string(self, action_str: str, action_dim: int) -> np.ndarray:
        """
        Parse action string to extract action bins.

        Args:
            action_str: String like "<action_bin_10><action_bin_20>..."

        Returns:
            Decoded action array
        """
        # Extract all action bin IDs
        action_bin_ids = re.findall(r"<action_bin_(\d+)>", action_str)
        action_bin_ids = np.array(list(map(int, action_bin_ids)))

        if len(action_bin_ids) > action_dim:
            action_bin_ids = action_bin_ids[:action_dim]
        elif len(action_bin_ids) < action_dim and len(action_bin_ids) > 0:
            action_bin_ids = np.concatenate(
                [
                    action_bin_ids,
                    [action_bin_ids[-1]] * (action_dim - len(action_bin_ids)),
                ]
            )
        elif len(action_bin_ids) == 0:
            action_bin_ids = [0] * action_dim

        # Convert action bin IDs to token IDs
        action_token_ids = []
        for bin_id in action_bin_ids:
            token_id = self.base_tokenizer.convert_tokens_to_ids(f"<action_bin_{bin_id}>")
            action_token_ids.append(token_id)

        # return self.decode_token_ids_to_actions(np.array(action_token_ids))
        return np.array(action_token_ids)

    def decode_token_ids_to_position(self, position_token_ids: np.ndarray) -> np.ndarray:
        """
        Decode position token IDs to continuous 3D positions.

        Args:
            position_token_ids: Token IDs of shape [3] or [B, 3]

        Returns:
            Continuous positions of shape [3] or [B, 3]
        """
        token_to_bin = {int(tid): i for i, tid in enumerate(self.position_token_ids)}
        shape = position_token_ids.shape
        flat_ids = position_token_ids.flatten()
        bin_indices = np.array([token_to_bin.get(int(tid), 0) for tid in flat_ids]).reshape(shape)
        bin_indices = np.clip(bin_indices, 0, len(self.position_bin_centers) - 1)
        return self.position_bin_centers[bin_indices]
