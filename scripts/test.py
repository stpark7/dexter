#!/usr/bin/env python3
"""
Test/evaluation script for Dexter model with hand metrics.

Usage:
    # Load config from checkpoint directory (recommended)
    python scripts/test.py \
        --checkpoint ./checkpoints/experiment-name/checkpoint-1000 \
        --batch-size 8 \
        --num-steps 10

    # Or specify configs manually
    python scripts/test.py \
        --checkpoint ./checkpoints/experiment-name/final \
        --data-config configs/data/dexgys.yaml \
        --output-dir test_output
"""

import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
from omegaconf import OmegaConf
from train import get_model_cls, get_model_config
from transformers import TrainingArguments

import dexter.models.loss as loss
from dexter.data import Collator, DexGYSDataset, DexonomyDataset
from dexter.data.rgbd_simulation import SimulatePartialRGBDObservation
from dexter.data.transforms import CompositeTransform, build_preprocess_and_postprocess_transforms
from dexter.training import CustomTrainer
from dexter.utils.logger import RankedLogger
from dexter.utils.shadowhand import ShadowHandModel

log = RankedLogger(__name__)


def main(
    checkpoint_dir: str,
    # data configs
    split: str = "test",
    batch_size: int = 24,
    data_dir: Optional[str] = None,
    training_stage: Optional[int] = None,
    num_workers: int = 4,
    # inference configs
    precision: str = "bfloat16",
    constrain_to_actions: bool = True,
    max_new_tokens: int = 256,
    output_dir: str = "test_output",
    parse_ecot: bool = True,
    save_pred: bool = False,
    measure_inference_time: bool = False,
    postfix_contact_string: str | None = None,
    steer_link_num: int | None = None,
    # partial observation configs
    partial_obs: bool = False,
    partial_obs_mode: str = "ego_thirdperson",  # "hemisphere" or "ego_thirdperson"
    partial_obs_num_views: int = 3,
    partial_obs_camera_radius: float = 0.3,
    partial_obs_seed: int | None = 42,
    # sensor noise configs
    partial_obs_add_noise: bool = False,
    partial_obs_depth_noise: float = 2.0,  # mm
    partial_obs_lateral_noise: float = 1.0,  # mm
    partial_obs_outlier_ratio: float = 1.0,  # %
):
    # ===== Setup output directory (before anything logs, so test.log captures it all) =====
    checkpoint_dir = Path(checkpoint_dir)
    experiment_name = checkpoint_dir.parent.stem
    model_name = checkpoint_dir.name
    postfix = f"_{split}"
    if postfix_contact_string is not None:
        postfix += f"_{postfix_contact_string}"
    if steer_link_num is not None:
        postfix += f"_steer{steer_link_num}"
    if partial_obs:
        if partial_obs_mode == "ego_thirdperson":
            postfix += f"_partial_{partial_obs_mode}"
        else:
            postfix += f"_partial_{partial_obs_mode}_v{partial_obs_num_views}"
        if partial_obs_add_noise:
            postfix += (
                f"_noise{partial_obs_depth_noise:.0f}mm_outlier{partial_obs_outlier_ratio:.0f}pct"
            )
    output_dir = Path(output_dir) / f"{experiment_name}_{model_name}{postfix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mirror console logging into the run dir so the whole run (incl. eval results) is persisted.
    file_handler = logging.FileHandler(output_dir / "test.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info(f"Output directory: {output_dir}")

    # Try to load training config from checkpoint directory
    config_path = os.path.join(os.path.normpath(checkpoint_dir), "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.normpath(checkpoint_dir)), "config.yaml")

    assert Path(config_path).exists(), f"No config.yaml found at {config_path}"
    training_cfg = OmegaConf.load(config_path)
    log.info(f"✓ Loaded training config from {config_path}")

    #  Override data config
    data_config = training_cfg.data
    data_config.split = split
    data_config.eval_transforms.tokenize.contact_position_dropout = 0.0
    if training_stage is not None:
        data_config.eval_transforms.tokenize.training_stage = training_stage
    if data_dir is not None:
        data_config.path = data_dir
    if postfix_contact_string is not None:
        data_config.eval_transforms.tokenize.postfix_contact_string = postfix_contact_string
    if steer_link_num is not None:
        data_config.eval_transforms.tokenize.steer_link_num = steer_link_num
    # Apply CLI overrides
    eval_batch_size = (
        batch_size if batch_size is not None else training_cfg.training.get("eval_batch_size", 8)
    )
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    log.info(f"Evaluation batch size: {eval_batch_size}")

    # ===== 1. Load Model =====
    log.info(f"Loading model from {checkpoint_dir}")
    model_cls = get_model_cls(training_cfg)
    model_config = get_model_config(training_cfg)
    model = model_cls(model_config)

    log.info("✓ Model loaded")

    preprocess_transforms, postprocess_transforms, base_tokenizer, action_tokenizer = (
        build_preprocess_and_postprocess_transforms(
            training_cfg.model, data_config, split=split, model=model
        )
    )

    # Add partial observation simulation if enabled
    if partial_obs:
        partial_obs_transform = SimulatePartialRGBDObservation(
            num_views=partial_obs_num_views,
            camera_radius=partial_obs_camera_radius,
            camera_mode=partial_obs_mode,
            seed=partial_obs_seed,
            add_noise=partial_obs_add_noise,
            depth_noise_std=partial_obs_depth_noise / 1000.0,  # mm to meters
            lateral_noise_std=partial_obs_lateral_noise / 1000.0,  # mm to meters
            outlier_ratio=partial_obs_outlier_ratio / 100.0,  # % to ratio
        )
        # Prepend partial observation transform (apply before other transforms)
        preprocess_transforms = CompositeTransform(
            transforms=[partial_obs_transform] + list(preprocess_transforms.transforms)
        )
        log.info(f"✓ Partial observation enabled: {partial_obs_transform}")

    # Load checkpoint weights into the model (which now has special tokens)
    log.info(f"Loading checkpoint weights from {checkpoint_dir}")
    checkpoint_file = checkpoint_dir / "model.safetensors"
    assert checkpoint_file.exists(), f"Checkpoint file not found at {checkpoint_file}"
    safetensors.torch.load_model(model, checkpoint_file, strict=False)
    log.info("✓ Loaded checkpoint from safetensors")

    if hasattr(model, "vlm") and hasattr(model.vlm, "tie_weights"):
        model.vlm.tie_weights()
        log.info("✓ Tied VLM weights (lm_head ↔ embed_tokens)")

    model.postprocess_transforms = postprocess_transforms
    model.base_tokenizer = base_tokenizer
    model.action_tokenizer = action_tokenizer
    model.eval()
    model.to(device)
    log.info("✓ Model ready for evaluation")

    if "dexonomy" in data_config.path:
        test_dataset = DexonomyDataset(
            data_path=data_config.path,
            split=split,
            transform=preprocess_transforms,
        )
    else:
        test_dataset = DexGYSDataset(
            data_path=data_config.path,
            split=split,
            transform=preprocess_transforms,
        )
    log.info(f"✓ Loaded {len(test_dataset)} test samples")

    # ===== 8. Setup Hand Model and Loss Functions =====
    log.info("Setting up hand model and loss functions...")
    hand_model = ShadowHandModel(base_dir="./assets/shadowhand", device=device)

    # Configure collator with max_points for partial observation batching
    # When using partial observation, point clouds have variable sizes
    # Setting max_points enables resampling to a fixed size for batched inference
    max_points = data_config.get("max_points", 10000) if partial_obs else None
    collator = Collator(max_points=max_points)
    model.to_precision(precision=precision)
    log.info(f"✓ Model precision: {precision}")

    loss_functions = {
        "hand_chamfer": loss.get_hand_chamfer_loss,
        "cmap": loss.get_cmap_loss,
        "obj_penetration": partial(loss.get_obj_penetration_loss, training=False),
        "self_penetration": partial(loss.get_self_penetration_loss, training=False),
    }
    log.info("✓ Hand model and loss functions initialized")

    # ===== 9. Setup Trainer for Evaluation =====
    # Create dummy training args (required by Trainer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=eval_batch_size,
        do_eval=True,
        do_train=False,
        bf16=precision == "bfloat16",
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        report_to="none",  # Disable wandb for testing
    )

    # Create custom trainer with hand metrics
    trainer = CustomTrainer(
        hand_model=hand_model,
        loss_functions=loss_functions if isinstance(test_dataset, DexGYSDataset) else None,
        save_predictions=save_pred,
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=collator,
        constrain_to_actions=constrain_to_actions,
        max_new_tokens=max_new_tokens,
        parse_ecot=parse_ecot,
        measure_inference_time=measure_inference_time,
    )

    log.info("\n" + "=" * 80)
    log.info("Starting evaluation with hand metrics")
    log.info("=" * 80 + "\n")

    metrics = trainer.evaluate()

    log.info("\n" + "=" * 80)
    log.info("Evaluation Results")
    log.info("=" * 80)
    for metric_name, value in sorted(metrics.items()):
        log.info(f"{metric_name}: {value:.6f}")
    log.info("=" * 80 + "\n")

    if save_pred and trainer.predictions:
        prediction_path = Path(output_dir) / "predictions.json"
        log.info(f"Saving {len(trainer.predictions)} predictions to {prediction_path}")

        predictions = []
        for i, pred_data in enumerate(trainer.predictions):
            predictions.append(
                {
                    "obj_id": pred_data["scene_name"],
                    "guidance": pred_data["prompt"],
                    "predictions": pred_data["pred"].tolist(),
                    "targets": pred_data["gt"].tolist(),
                    "contact": pred_data.get("contact", None),
                }
            )

        with open(prediction_path, "w") as f:
            json.dump(predictions, f, indent=4)
        log.info(f"✓ Predictions saved to {prediction_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
