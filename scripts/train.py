#!/usr/bin/env python3
"""
Training script for Dexter models with Hydra config.

Usage:
    # Basic training with default config
    python scripts/train.py

    # Use specific model/data config
    python scripts/train.py model=dexter_qwen3 data=dexgys

    # Override specific parameters
    python scripts/train.py model.encoder_type=partfield training.batch_size=32

    # Hyperparameter sweep (multi-run)
    python scripts/train.py -m training.learning_rate=1e-4,5e-5,1e-5
"""

import os
from pathlib import Path

import hydra
import safetensors.torch
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from dexter.data import Collator, DexGYSDataset, DexonomyDataset
from dexter.data.transforms import build_preprocess_and_postprocess_transforms
from dexter.training import CustomTrainer
from dexter.utils import RankedLogger, is_main_process

log = RankedLogger(__name__, rank_zero_only=True)


def get_model_cls(cfg: DictConfig):
    model_type = cfg.model.model_type
    if "qwen" in model_type:
        from dexter.models import DexterQwenForActionPrediction

        return DexterQwenForActionPrediction
    elif "smollm" in model_type:
        from dexter.models import DexterSmolLM2ForActionPrediction

        return DexterSmolLM2ForActionPrediction
    else:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")


def get_model_config(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    model_type = cfg.model.model_type
    if "qwen" in model_type:
        from dexter.models import DexterQwenConfig

        return DexterQwenConfig(**cfg_dict)
    elif "smollm" in model_type:
        from dexter.models import DexterSmolLM2Config

        return DexterSmolLM2Config(**cfg_dict)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg.training.output_dir = os.path.join("checkpoints", cfg.experiment_name)

    # Save Hydra config to checkpoint directory for reproducibility
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.training.output_dir, "config.yaml")
    OmegaConf.save(cfg, config_save_path)
    log.info(f"✓ Saved training config to {config_save_path}")

    # Print config for debugging
    log.info("=" * 80)
    log.info("Training Configuration:")
    log.info("=" * 80)
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=" * 80)

    # Initialize wandb with Hydra config
    # The Trainer will detect this and use the existing run
    if is_main_process() and cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    # ===== 1. Create Model =====
    model_cls = get_model_cls(cfg)
    model_config = get_model_config(cfg)
    if hasattr(cfg, "resume_checkpoint") and cfg.resume_checkpoint:
        model = model_cls.from_pretrained(cfg.resume_checkpoint)
    else:
        model = model_cls(model_config)

        # Setup special tokens and resize embeddings BEFORE loading checkpoint
        # This is required because the checkpoint may have been saved with expanded vocab
        if hasattr(model, "setup_special_tokens"):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(cfg.model.tokenizer.model_id)
            base_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            model.setup_special_tokens(
                base_tokenizer=base_tokenizer,
                n_action_bins=cfg.model.action.bins,
                n_position_bins=cfg.model.action.get("position_bins", 256),
            )
            log.info("✓ Setup special tokens and resized embeddings before loading checkpoint")

        # Load pretrained weights if specified
        if cfg.training.get("pretrained_checkpoint", None):
            pretrained_checkpoint = cfg.training.pretrained_checkpoint
            log.info(f"Loading pretrained weights from {pretrained_checkpoint}")
            if pretrained_checkpoint.endswith(".safetensors"):
                missing, unexpected = safetensors.torch.load_model(
                    (
                        model.module
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                        else model
                    ),
                    pretrained_checkpoint,
                    strict=False,
                )
                if missing and is_main_process():
                    log.warning(f"Missing keys: {missing}\n")
                if unexpected and is_main_process():
                    log.warning(f"Unexpected keys: {unexpected}\n")
            else:
                model = model.from_pretrained(pretrained_checkpoint)

    # ===== 3. Build Transforms =====
    preprocess_transforms, postprocess_transforms, base_tokenizer, action_tokenizer = (
        build_preprocess_and_postprocess_transforms(cfg.model, cfg.data, model=model)
    )

    if cfg.data.name.startswith("dexgys"):
        dataset_cls = DexGYSDataset
    elif cfg.data.name.startswith("dexonomy"):
        dataset_cls = DexonomyDataset
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.name}")

    train_dataset = dataset_cls(
        data_path=cfg.data.path,
        split="train",
        transform=preprocess_transforms,
        overfitting=cfg.data.overfitting,
    )
    log.info(f"✓ Loaded {len(train_dataset)} training samples")

    # Connect tokenizers for debugging
    model.postprocess_transforms = postprocess_transforms
    model.base_tokenizer = base_tokenizer
    model.action_tokenizer = action_tokenizer

    model.to_precision(precision=cfg.training.precision)
    log.info(f"✓ Training precision: {cfg.training.precision}")

    # ===== 6. Setup Training Arguments =====
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        # Training
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.get("eval_batch_size", 4),
        gradient_accumulation_steps=cfg.training.get("grad_accum_steps", 1),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        # Optimization
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        adam_beta1=cfg.training.b1,
        adam_beta2=cfg.training.b2,
        adam_epsilon=cfg.training.eps,
        # Precision
        fp16=cfg.training.get("fp16", False),
        bf16=cfg.training.get("bf16", True),
        # Logging & Saving
        logging_steps=cfg.training.log_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=3,
        do_eval=cfg.training.do_eval,
        eval_strategy="no",
        eval_steps=None,
        # Reporting
        report_to="wandb" if cfg.wandb.enabled else "none",
        run_name=cfg.experiment_name,
        # Device
        dataloader_num_workers=cfg.training.get("num_workers", 4),
        dataloader_pin_memory=True,
        # Misc
        remove_unused_columns=False,  # Keep all data fields
        ddp_find_unused_parameters=True,
    )

    # Create collator for autoregressive mode
    # Pass max_points for partial observation training (variable-size point clouds)
    max_points = cfg.data.get("max_points", None)
    collator = Collator(max_points=max_points)

    # ===== 7. Create Trainer =====
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # ===== 8. Train =====
    log.info("\n" + "=" * 80)
    log.info(f"Starting training: {cfg.experiment_name}")
    log.info("=" * 80 + "\n")
    trainer.train()

    # ===== 9. Save Final Model =====
    final_dir = Path(cfg.training.output_dir) / "final"
    model.save_pretrained(final_dir)
    log.info(f"\n✓ Model saved to {final_dir}")

    # Trainer automatically handles wandb.finish() when done
    if is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()
