"""Shared base class for modern autoregressive Dexter model variants.

Provides the full training forward pass, prefill+generate inference pattern,
and action/ECoT parsing. Subclasses only implement `_init_vlm()` and
the `language_model` property.
"""

import dataclasses
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    PreTrainedModel,
    Qwen2ForCausalLM,
    Qwen3VLForConditionalGeneration,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_outputs import ModelOutput

from dexter.models.action_tokenizer import setup_special_tokens_and_resize_embeddings
from dexter.models.loss import (
    compute_actions_l1_loss,
    compute_category_token_accuracies,
    compute_positions_l1_loss,
)
from dexter.utils.logger import RankedLogger

from .configuration_dexter import (
    DexterQwenConfig,
    DexterSmolLM2Config,
)
from .generation_dexter import GraspGrammarLogitsProcessor

log = RankedLogger(__name__, rank_zero_only=True)


class PointCloudProjector(nn.Module):
    def __init__(self, pc_embed_dim: int, hidden_size: int):
        super().__init__()
        self.pc_embed_dim = pc_embed_dim
        self.hidden_size = hidden_size

        self.layernorm = nn.LayerNorm(pc_embed_dim)
        self.proj = nn.Linear(pc_embed_dim, hidden_size, bias=False)
        self.activation = nn.GELU()

        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, pc_features: torch.Tensor) -> torch.Tensor:
        pc_features = self.layernorm(pc_features)
        pc_features = self.proj(pc_features)
        return self.activation(pc_features)


@dataclasses.dataclass
class DexterOutput(ModelOutput):
    """Output type of modern autoregressive Dexter models."""

    loss: torch.FloatTensor = None
    token_accuracy_overall: torch.FloatTensor = None
    token_accuracy_action: torch.FloatTensor = None
    token_accuracy_joint: torch.FloatTensor = None
    token_accuracy_position: torch.FloatTensor = None
    action_l1_loss: torch.FloatTensor = None
    position_l1_loss: torch.FloatTensor = None


class DexterForActionPrediction(PreTrainedModel):
    """Shared base for the backbone-specific autoregressive Dexter variants.

    Backbone subclasses (defined below) must:
      1. Set `config_class` and `base_model_prefix`.
      2. Override `_init_vlm(config)` → returns vlm_hidden_size (int).
      3. Override the `language_model` property → returns the core transformer
         (the object whose `.embed_tokens` gives the token embedding layer).
    """

    supports_gradient_checkpointing = True
    main_input_name = "tokenized_prompt"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.pc_encoder = None
        self.pc_proj = None
        self.vlm = None

        # Point cloud encoder (shared)
        self._init_pc_encoder(config)

        # VLM (subclass-specific)
        vlm_hidden_size = self._init_vlm(config)

        # Point cloud projector (shared)
        self.pc_proj = PointCloudProjector(self.pc_encoder.point_feat_dim, vlm_hidden_size)

        # Freeze parameters if requested
        self._freeze_if_requested(config)

        # Action tokenizer will be set externally
        self.action_tokenizer = None

        self.post_init()

        # Ensure VLM weights are properly tied (lm_head <-> embed_tokens)
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_pc_encoder(self, config):
        if config.encoder.type == "uni3d":
            from dexter.models.encoders.uni3d import Uni3D

            self.pc_encoder = Uni3D.from_pretrained(config.encoder)
        elif config.encoder.type == "partfield":
            from dexter.models.encoders.partfield import PartField

            self.pc_encoder = PartField.from_pretrained(config.encoder)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder.type}")

    def _init_vlm(self, config) -> int:
        """Load the VLM, store as ``self.vlm``, return hidden_size."""
        raise NotImplementedError

    def _freeze_if_requested(self, config):
        if config.encoder.freeze:
            log.info("Freezing PC encoder parameters")
            for param in self.pc_encoder.parameters():
                param.requires_grad = False
        if config.vlm.freeze:
            log.info("Freezing VLM parameters")
            for param in self.vlm.parameters():
                param.requires_grad = False

    @property
    def language_model(self):
        """Return the core transformer backbone (has ``.embed_tokens``)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Precision / attention-mask utilities
    # ------------------------------------------------------------------

    def to_precision(
        self,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        extra_params_to_keep_float32: list[str] = [],
    ):
        assert precision in ["bfloat16", "float32"], f"Invalid precision: {precision}"

        if precision == "float32":
            self.to(dtype=torch.float32)
            return

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "layernorm",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        if len(extra_params_to_keep_float32) > 0:
            params_to_keep_float32.extend(extra_params_to_keep_float32)

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)
            else:
                param.data = param.data.to(dtype=torch.bfloat16)
        for name, buffer in self.named_buffers():
            if any(selector in name for selector in params_to_keep_float32):
                buffer.data = buffer.data.to(dtype=torch.float32)
            else:
                buffer.data = buffer.data.to(dtype=torch.bfloat16)

    def make_att_2d_masks(self, pad_masks, att_masks):
        """Create 2D causal attention masks from 1D masks.

        Args:
            pad_masks: bool[B, N] - True if part of input, False if padding
            att_masks: int32[B, N] - 0 for full attention, 1 for causal attention

        Returns:
            2D attention mask where True means can attend, False means mask out
        """
        if att_masks.ndim != 2:
            raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
        if pad_masks.ndim != 2:
            raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")

        # Cumulative sum creates groups: tokens with same cumsum can attend to each other
        cumsum = torch.cumsum(att_masks, dim=1)
        # Token i can attend to token j if cumsum[j] <= cumsum[i]
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        # Also respect padding
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def make_att_4d_masks(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def setup_special_tokens(self, base_tokenizer, n_action_bins: int, n_position_bins: int):
        log.info("Setting up special tokens and resizing embeddings...")
        setup_special_tokens_and_resize_embeddings(
            tokenizer=base_tokenizer,
            model=self.vlm,
            n_action_bins=n_action_bins,
            n_position_bins=n_position_bins,
        )
        log.info("Special tokens setup complete")

    # ------------------------------------------------------------------
    # Multimodal input preparation
    # ------------------------------------------------------------------

    def prepare_multimodal_inputs(
        self,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        labels: torch.Tensor,
        pc_embeddings: torch.Tensor,
        vision_token_id: int,
    ):
        # labels may be None at inference (only needed for the training loss).
        if isinstance(vision_token_id, list):
            vision_token_id = vision_token_id[0]

        vision_mask = tokenized_prompt == vision_token_id
        idx_row, idx_col = torch.where(vision_mask)
        assert (idx_col == idx_col[0]).all(), "Vision tokens are not in the same column"
        idx_col = idx_col[0].item()

        tokenized_prompt_interpolated = torch.cat(
            [
                tokenized_prompt[:, :idx_col],
                tokenized_prompt[:, idx_col : (idx_col + 1)].repeat(1, pc_embeddings.shape[1]),
                tokenized_prompt[:, (idx_col + 1) :],
            ],
            dim=1,
        )
        tokenized_prompt_mask_interpolated = torch.cat(
            [
                tokenized_prompt_mask[:, :idx_col],
                tokenized_prompt_mask[:, idx_col : (idx_col + 1)].repeat(1, pc_embeddings.shape[1]),
                tokenized_prompt_mask[:, (idx_col + 1) :],
            ],
            dim=1,
        )
        labels_interpolated = None
        if labels is not None:
            labels_interpolated = torch.cat(
                [
                    labels[:, :idx_col],
                    labels[:, idx_col : (idx_col + 1)].repeat(1, pc_embeddings.shape[1]),
                    labels[:, (idx_col + 1) :],
                ],
                dim=1,
            )
        return (
            tokenized_prompt_interpolated,
            tokenized_prompt_mask_interpolated,
            labels_interpolated,
        )

    # ------------------------------------------------------------------
    # Embed prefix (vision token replacement)
    # ------------------------------------------------------------------

    def embed_prefix(
        self,
        pointclouds,
        input_ids,
        attention_mask,
        labels=None,
        vision_token_id: int = None,
    ):
        assert vision_token_id is not None, "Vision token ID is required"
        if isinstance(vision_token_id, list):
            vision_token_id = vision_token_id[0]

        # Encode point cloud
        pc_embeddings = self.pc_encoder(
            pointclouds,
            return_point_features=False,
        )

        pc_embeddings = pc_embeddings.to(self.pc_proj.proj.weight.dtype)
        pc_embeddings = self.pc_proj(pc_embeddings)

        pc_embeddings = pc_embeddings.to(self.language_model.embed_tokens.weight.dtype)
        batch_size, num_pc_embs = pc_embeddings.shape[:2]

        input_ids_interpolated, attention_mask_interpolated, labels_interpolated = (
            self.prepare_multimodal_inputs(
                input_ids,
                attention_mask,
                labels,
                pc_embeddings,
                vision_token_id,
            )
        )

        # Get text embeddings
        text_embeddings = self.language_model.embed_tokens(input_ids_interpolated)

        # Replace vision tokens with PC embeddings
        vision_mask = input_ids_interpolated == vision_token_id
        prefix_embeddings = text_embeddings.clone()
        vision_mask_expanded = vision_mask.unsqueeze(-1).expand_as(prefix_embeddings)
        prefix_embeddings.masked_scatter_(vision_mask_expanded, pc_embeddings)

        # Create attention scope masks
        prefix_attention_mask = torch.ones_like(input_ids_interpolated)
        prefix_attention_mask.masked_fill_(vision_mask, 0)

        return (
            prefix_embeddings,
            attention_mask_interpolated,
            prefix_attention_mask,
            num_pc_embs,
            input_ids_interpolated,
            labels_interpolated,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pointcloud: torch.Tensor,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        actions: torch.Tensor = None,
        prompts: list[str] = None,
        labels: torch.Tensor = None,
        masks: torch.Tensor = None,
        original_action_dim: int = None,
        noise: torch.Tensor = None,
        time: torch.Tensor = None,
        return_dict: bool = True,
        vision_token_id: int = None,
        *args,
        **kwargs,
    ):
        if vision_token_id is None:
            vision_token_id = kwargs.get("vision_token_id", None)

        (
            inputs_embeds,
            pad_masks,
            att_masks,
            num_pc_embs,
            input_ids_interpolated,
            labels_interpolated,
        ) = self.embed_prefix(
            pointcloud, tokenized_prompt, tokenized_prompt_mask, labels, vision_token_id
        )

        att_masks_2d = self.make_att_2d_masks(pad_masks, att_masks)
        att_masks_2d = att_masks_2d.bool()
        att_masks_4d = self.make_att_4d_masks(att_masks_2d)
        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

        outputs_embeds = self.vlm.model(
            input_ids=None,
            attention_mask=att_masks_4d,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        ).last_hidden_state

        logits = self.vlm.lm_head(outputs_embeds)

        outputs = {}

        if labels_interpolated is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels_interpolated[:, 1:].contiguous()
            shift_input_ids = input_ids_interpolated[:, 1:].contiguous()

            pred_logit_ids = shift_logits.argmax(dim=-1)

            valid_token_mask = shift_labels != -100

            category_accuracies = compute_category_token_accuracies(
                self.action_tokenizer,
                pred_logit_ids,
                shift_input_ids,
                overall_mask=valid_token_mask,
            )
            outputs["token_accuracy_overall"] = category_accuracies["overall"]
            outputs["token_accuracy_action"] = category_accuracies["action"]
            outputs["token_accuracy_joint"] = category_accuracies["joint"]
            outputs["token_accuracy_position"] = category_accuracies["position"]

            action_l1_loss = compute_actions_l1_loss(
                self.action_tokenizer, pred_logit_ids, shift_labels
            )
            outputs["action_l1_loss"] = action_l1_loss
            position_l1_loss = compute_positions_l1_loss(
                self.action_tokenizer, pred_logit_ids, shift_labels
            )
            outputs["position_l1_loss"] = position_l1_loss

            ce_kwargs = {}
            if hasattr(self.config, "label_smoothing") and self.config.label_smoothing is not None:
                ce_kwargs["label_smoothing"] = self.config.label_smoothing
            loss_fct = nn.CrossEntropyLoss(reduction="mean", **ce_kwargs)

            vocab_size = logits.shape[-1]
            shift_logits_flat = shift_logits.view(-1, vocab_size)
            shift_labels_flat = shift_labels.view(-1)

            outputs["loss"] = loss_fct(shift_logits_flat, shift_labels_flat)

        return DexterOutput(loss=outputs.pop("loss"), **outputs)

    # ------------------------------------------------------------------
    # Inference: prefill → generate → parse
    # ------------------------------------------------------------------

    def _prefill(self, pointclouds, input_ids, attention_mask, vision_token_id=None):
        (
            inputs_embeds,
            pad_masks,
            att_masks,
            num_pc_embs,
            input_ids_interpolated,
            _labels_interpolated,
        ) = self.embed_prefix(
            pointclouds, input_ids, attention_mask, vision_token_id=vision_token_id
        )

        att_masks_2d = self.make_att_2d_masks(pad_masks, att_masks)
        att_masks_4d = self.make_att_4d_masks(att_masks_2d.bool())
        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

        outputs = self.vlm.model(
            input_ids=None,
            attention_mask=att_masks_4d,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=None,
            return_dict=True,
        )

        valid_lengths = pad_masks.sum(dim=1)
        batch_indices = torch.arange(pad_masks.shape[0], device=pad_masks.device)
        last_valid_positions = valid_lengths - 1
        last_hidden = outputs.last_hidden_state[batch_indices, last_valid_positions].unsqueeze(1)
        first_token_logits = self.vlm.lm_head(last_hidden).squeeze(1)

        return {
            "past_key_values": outputs.past_key_values,
            "first_token_logits": first_token_logits,
            "pad_masks": pad_masks,
            "prefix_seq_len": pad_masks.shape[1],
        }

    def _generate_tokens(
        self,
        prefill_outputs,
        max_new_tokens,
        temperature=0.0,
        top_k=None,
        constrain_to_actions=True,
    ):
        past_key_values = prefill_outputs["past_key_values"]
        first_token_logits = prefill_outputs["first_token_logits"]
        pad_masks = prefill_outputs["pad_masks"]
        prefix_seq_len = prefill_outputs["prefix_seq_len"]

        batch_size = pad_masks.shape[0]
        device = pad_masks.device

        if temperature == 0.0:
            first_token = first_token_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(first_token_logits / temperature, dim=-1)
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                first_token = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                first_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        if max_new_tokens <= 1:
            return first_token.unsqueeze(1)

        logits_processor = None
        if constrain_to_actions:
            processor = GraspGrammarLogitsProcessor.from_grasp_tokenizer(
                self.action_tokenizer,
                action_dim=self.config.action_dim,
            )
            logits_processor = LogitsProcessorList([processor])

        attention_mask = torch.cat(
            [
                pad_masks.long(),
                torch.ones(batch_size, 1, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        gen_kwargs = dict(
            input_ids=first_token.unsqueeze(1),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=torch.tensor([prefix_seq_len], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens - 1,
            eos_token_id=self.base_tokenizer.eos_token_id,
            pad_token_id=self.base_tokenizer.eos_token_id,
        )

        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k

        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor

        generated_ids = self.vlm.generate(**gen_kwargs)
        return generated_ids

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_generated(self, generated_token_ids, action_dim, return_cot=False):
        """Parse generated tokens into actions (and optionally contact reasoning).

        Routes through ``action_tokenizer.parse_ecot_output`` (the single source of
        truth for the ``<|action_start|>…<|action_end|>`` and
        ``<|joint_start|>…<|joint_end|>`` regions). With ``return_cot=False`` this is
        byte-for-byte equivalent to the former ``_parse_generated_actions``; with
        ``return_cot=True`` it additionally returns per-joint contact positions
        (former ``_parse_generated_ecot``).
        """
        action_preds = []
        contact_preds = []
        for ids in generated_token_ids:
            parsed = self.action_tokenizer.parse_ecot_output(
                ids.cpu().numpy().tolist(), action_dim=action_dim
            )
            result = self.postprocess_transforms({"tokenized_actions": parsed["predicted_action"]})
            action_preds.append(result["actions"])
            if return_cot:
                contact_preds.append(
                    {j: np.array(p) for j, p in parsed["contact_reasoning"].items()}
                )

        actions = np.stack(action_preds, axis=0)
        return {"actions": actions, "contacts": contact_preds} if return_cot else actions

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        pointclouds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = None,
        temperature: float = 0.0,
        top_k: int = None,
        constrain_to_actions: bool = True,
        return_cot: bool = False,
        vision_token_id: int = None,
        *args,
        **kwargs,
    ):
        """Autoregressively decode grasp actions.

        Decodes via HuggingFace ``generate()`` (``_generate_tokens``), shared by the
        direct and ECoT modes. ``return_cot=False`` returns an ``[B, action_dim]``
        array; ``return_cot=True`` returns ``{"actions", "contacts"}`` with the
        per-joint contact reasoning. When ``constrain_to_actions`` is set, a
        :class:`GraspGrammarLogitsProcessor` enforces the grasp grammar (counted
        ``<|action_start|>…<|action_end|>`` blocks and, for ECoT, the
        ``<|joint_start|>…<|joint_end|>`` contact block) — orthogonal to ``return_cot``.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.action_horizon * self.config.action_dim
        if vision_token_id is None:
            vision_token_id = kwargs.get("vision_token_id", None)

        prefill = self._prefill(pointclouds, input_ids, attention_mask, vision_token_id)
        generated_ids = self._generate_tokens(
            prefill, max_new_tokens, temperature, top_k, constrain_to_actions
        )
        return self._parse_generated(generated_ids, self.config.action_dim, return_cot=return_cot)


# ======================================================================
# Backbone-specific variants
# ======================================================================


class DexterQwenForActionPrediction(DexterForActionPrediction):
    """Dexter AR model with a Qwen2 (text-only) or Qwen3-VL (multimodal) backbone.

    When using Qwen3-VL, the vision tower is removed since we use a PC encoder.
    """

    config_class = DexterQwenConfig
    base_model_prefix = "dexter_qwen"

    def _init_vlm(self, config) -> int:
        log.info(f"Loading Qwen model from: {config.vlm.model_id}")

        if config.vlm.model_id.startswith("Qwen/Qwen2"):
            self.vlm = Qwen2ForCausalLM.from_pretrained(
                config.vlm.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=config.attention_type,
            )
            vlm_hidden_size = self.vlm.config.hidden_size

        elif config.vlm.model_id.startswith("Qwen/Qwen3-VL"):
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=config.attention_type,
            )
            vlm_hidden_size = self.vlm.config.text_config.hidden_size

            # Delete vision tower and visual projector (we use PC encoder instead)
            for attr_name in ["_visual", "visual"]:
                for obj in [self.vlm, self.vlm.model]:
                    if hasattr(obj, attr_name):
                        try:
                            delattr(obj, attr_name)
                            log.info(f"Removed {type(obj).__name__}.{attr_name}")
                        except (AttributeError, TypeError):
                            try:
                                setattr(obj, attr_name, None)
                                log.info(f"Set {type(obj).__name__}.{attr_name} to None")
                            except AttributeError:
                                log.warning(f"Could not remove {type(obj).__name__}.{attr_name}")
        else:
            raise ValueError(f"Unknown VLM model ID: {config.vlm.model_id}")

        return vlm_hidden_size

    @property
    def language_model(self):
        if isinstance(self.vlm, Qwen3VLForConditionalGeneration):
            return self.vlm.language_model
        elif isinstance(self.vlm, Qwen2ForCausalLM):
            return self.vlm.model
        else:
            raise ValueError(f"Unknown VLM model: {self.vlm}")


class DexterSmolLM2ForActionPrediction(DexterForActionPrediction):
    """Dexter AR model with a SmolLM2 (Llama architecture) backbone."""

    config_class = DexterSmolLM2Config
    base_model_prefix = "dexter_smollm2"

    def _init_vlm(self, config) -> int:
        log.info(f"Loading SmolLM2 model from: {config.vlm.model_id}")
        self.vlm = LlamaForCausalLM.from_pretrained(
            config.vlm.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.attention_type,
        )
        return self.vlm.config.hidden_size

    @property
    def language_model(self):
        return self.vlm.model
