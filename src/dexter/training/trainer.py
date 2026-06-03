"""Custom Trainer for Dexter."""

import sys
import time

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import EvalLoopOutput

from dexter.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Hand metrics that are still computed and aggregated into the final results, but
# kept out of the live per-step progress display to reduce noise.
_LIVE_HIDDEN_METRICS = {"obj_penetration", "self_penetration"}

# Each scalar metric is its own "label value" segment. (log_key, label, format).
_SCALAR_METRICS = [
    ("epoch", "epoch", ".2f"),
    ("loss", "loss", ".4f"),
    ("learning_rate", "lr", ".2e"),
    ("grad_norm", "grad_norm", ".3f"),
]
# Related metrics share one "group a=.. b=.." segment. (group, format, members).
_METRIC_GROUPS = [
    (
        "acc",
        ".3f",
        [
            ("token_accuracy_overall", "all"),
            ("token_accuracy_action", "act"),
            ("token_accuracy_joint", "jnt"),
            ("token_accuracy_position", "pos"),
        ],
    ),
    (
        "l1",
        ".4f",
        [
            ("action_l1_loss", "act"),
            ("position_l1_loss", "pos"),
        ],
    ),
]


def format_log_metrics(step: int, logs: dict) -> str:
    """Render a Trainer log dict as a compact, aligned one-liner.

    Known training metrics (loss, lr, token accuracies, L1) are rendered first
    with friendly labels; any other numeric keys (e.g. eval_*) are appended
    as-is so nothing is silently dropped.
    """
    parts = [f"step {step}"]
    shown = set()

    for key, label, fmt in _SCALAR_METRICS:
        if key in logs:
            parts.append(f"{label} {logs[key]:{fmt}}")
            shown.add(key)

    for group, fmt, members in _METRIC_GROUPS:
        segment = [f"{label}={logs[key]:{fmt}}" for key, label in members if key in logs]
        if segment:
            parts.append(f"{group} " + " ".join(segment))
            shown.update(key for key, _ in members)

    for key, value in logs.items():
        if key not in shown and key != "total_flos" and isinstance(value, (int, float)):
            parts.append(f"{key} {value:.4g}")

    return " | ".join(parts)


class _FormattedLogMixin:
    """Override on_log to emit a formatted metrics line instead of the raw dict.

    Writes through the tqdm progress bar when one is active (so the line does
    not clobber the bar); otherwise prints.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        msg = format_log_metrics(state.global_step, logs)
        bar = getattr(self, "training_bar", None)
        if bar is not None:
            bar.write(msg)
        else:
            print(msg)


class FormattedProgressCallback(_FormattedLogMixin, ProgressCallback):
    """ProgressCallback that writes a formatted metrics line instead of the raw dict."""


class FormattedPrinterCallback(_FormattedLogMixin, PrinterCallback):
    """PrinterCallback variant used when tqdm is disabled."""


class CustomTrainer(Trainer):
    """
    Custom trainer for Dexter.

    Extends HuggingFace Trainer with:
    - Custom logging for action prediction metrics
    - Hand model evaluation with specialized metrics
    """

    def __init__(
        self,
        hand_model=None,
        loss_functions=None,
        num_inference_steps=10,
        save_predictions=False,
        constrain_to_actions=False,
        max_new_tokens=256,
        parse_ecot=False,
        measure_inference_time=False,
        *args,
        **kwargs,
    ):
        """
        Initialize Dexter trainer.

        Args:
            hand_model: ShadowHandModel for computing hand-based metrics
            loss_functions: Dict of loss function names to callables
            unnormalize: Unnormalize transform for actions
            num_inference_steps: Number of denoising steps for inference
            save_predictions: Whether to save predictions during evaluation
            measure_inference_time: Whether to measure per-sample inference time (adds overhead)
            *args, **kwargs: Passed to parent Trainer
        """
        super().__init__(*args, **kwargs)
        self.hand_model = hand_model
        self.loss_functions = loss_functions or {}
        self.num_inference_steps = num_inference_steps
        self.save_predictions = save_predictions
        self.predictions = []
        self._static_graph_set = False  # Track if we've set static graph
        self.constrain_to_actions = constrain_to_actions
        self.max_new_tokens = max_new_tokens
        self.parse_ecot = parse_ecot
        self.measure_inference_time = measure_inference_time

        # Replace the default log printer (which dumps the raw metrics dict) with a
        # formatted one, keeping the tqdm progress bar when it is enabled.
        if self.args.disable_tqdm:
            self.remove_callback(PrinterCallback)
            self.add_callback(FormattedPrinterCallback)
        else:
            self.remove_callback(ProgressCallback)
            self.add_callback(FormattedProgressCallback)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss and extract additional metrics from model output.

        Also handles setting static graph for DDP on the first training step.
        """
        # Set static graph once on first training step (after DDP wrapping is complete)
        if (
            not self._static_graph_set
            and isinstance(model, torch.nn.parallel.DistributedDataParallel)
            and self.args.gradient_checkpointing
            and self.args.ddp_find_unused_parameters
        ):
            log.info(
                "✓ Setting static graph for DDP optimization (gradient_checkpointing + find_unused_parameters)"
            )
            model._set_static_graph()
            self._static_graph_set = True

        outputs = model(**inputs)
        loss = outputs.loss

        # Store additional metrics for logging
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            log_metrics = {k: v.detach().item() for k, v in outputs.items() if v is not None}
            self.log(log_metrics)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for evaluation.

        Args:
            model: model
            inputs: Batch of inputs
            prediction_loss_only: Only return loss
            ignore_keys: Keys to ignore in outputs

        Returns:
            (loss, predictions, labels)
        """
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss

            # Optionally generate actions for metrics
            if not prediction_loss_only:
                generated_actions = model.sample_actions(
                    pointclouds=inputs["pointclouds"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_steps=self.num_inference_steps,
                )
                return (loss, generated_actions, inputs["actions"])

            return (loss, None, None)

    @staticmethod
    def _format_live_metrics(all_hand_metrics):
        """Running mean of the visible hand metrics, as a compact display string."""
        parts = []
        for name, values in all_hand_metrics.items():
            if name in _LIVE_HIDDEN_METRICS or not values:
                continue
            running = torch.cat(values).mean().item()
            parts.append(f"{name}={running:.3f}")
        return "  ".join(parts)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Custom evaluation loop with hand model metrics.

        Computes both standard loss and hand-specific metrics like:
        - hand_chamfer: Chamfer distance between predicted/target hand meshes
        - cmap: Contact map similarity
        - obj_penetration: Object penetration penalty
        - self_penetration: Self-penetration penalty
        """
        model = self.model
        model.eval()

        # Run standard evaluation first to get training loss
        if prediction_loss_only is None:
            prediction_loss_only = self.args.prediction_loss_only

        num_samples = 0

        # Hand model metrics (only if hand_model is available)
        all_hand_metrics = {key: [] for key in self.loss_functions} if self.hand_model else {}

        # Track per-sample inference times
        inference_times_per_sample = []

        # Clear predictions list if saving
        if self.save_predictions:
            self.predictions = []

        # Live progress bar (rich) on the main process when attached to a TTY;
        # otherwise fall back to a concise periodic log so redirected runs stay readable.
        use_rich = self.is_local_process_zero() and sys.stdout.isatty()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[metrics]}"),
            disable=not use_rich,
        )
        progress.start()
        eval_task = progress.add_task(
            description or "Evaluating", total=len(dataloader), metrics=""
        )

        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                # Compute hand metrics if hand_model is available
                num_samples += inputs["pointcloud"].shape[0]

                # Generate actions with autocast (matches training behavior)
                # Measure model inference time (only if enabled)
                if self.measure_inference_time:
                    torch.cuda.synchronize()
                    inference_start = time.perf_counter()

                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    preds = model.sample_actions(
                        pointclouds=inputs["pointcloud"],
                        input_ids=inputs["tokenized_prompt"],
                        attention_mask=inputs["tokenized_prompt_mask"],
                        vision_token_id=inputs["vision_token_id"],
                        num_steps=self.num_inference_steps,
                        constrain_to_actions=self.constrain_to_actions,
                        max_new_tokens=self.max_new_tokens,
                        return_cot=self.parse_ecot,
                    )
                if self.parse_ecot:
                    action_pred = preds["actions"]
                    contact_pred = preds["contacts"]
                else:
                    action_pred = preds

                if self.measure_inference_time:
                    torch.cuda.synchronize()
                    inference_end = time.perf_counter()
                    batch_inference_time = inference_end - inference_start
                    batch_size = inputs["pointcloud"].shape[0]
                    per_sample_time = batch_inference_time / batch_size
                    inference_times_per_sample.extend([per_sample_time] * batch_size)

                # Get ground truth actions
                action_pred = torch.from_numpy(action_pred).to(inputs["actions"].device)
                action_gt = inputs["actions"]
                action_pred = action_pred[..., : action_gt.shape[-1]]

                # Handle action_horizon: use first timestep for evaluation
                if action_pred.dim() == 3:  # [B, H, D]
                    action_pred = action_pred[:, 0, :]

                if action_gt.dim() == 3:  # [B, H, D]
                    action_gt = action_gt[:, 0, :]

                batch_size = action_pred.size(0)
                current_predictions = []
                pointcloud = inputs["pointcloud"]
                prompts = inputs.get("prompt_original", None)
                if not prompts:
                    prompts = inputs.get("prompt", None)
                for i in range(batch_size):
                    save_dict = {
                        "pred": action_pred[i].cpu().numpy(),
                        "gt": action_gt[i].cpu().numpy(),
                        "prompt": prompts[i],
                        "scene_name": inputs["scene_name"][i],
                        "pointcloud": pointcloud[i].cpu().numpy(),
                    }
                    if self.parse_ecot:
                        save_dict["contact"] = {
                            k: v.squeeze().tolist() for k, v in contact_pred[i].items()
                        }
                    current_predictions.append(save_dict)

                if self.hand_model and self.loss_functions:
                    # Extract XYZ from pointcloud (first 3 channels)
                    pointcloud = inputs["pointcloud"]
                    obj_pc_xyz = pointcloud[..., :3]

                    # Compute hand configurations
                    hand_gt = self.hand_model(
                        action_gt,
                        obj_pc_xyz,
                        with_meshes=True,
                        with_penetration=True,
                        with_surface_points=True,
                        with_penetration_keypoints=True,
                    )
                    hand_gt["obj_pc"] = obj_pc_xyz

                    hand_pred = self.hand_model(
                        action_pred,
                        obj_pc_xyz,
                        with_meshes=True,
                        with_penetration=True,
                        with_surface_points=True,
                        with_penetration_keypoints=True,
                    )

                    # Compute hand-specific losses
                    for loss_name, loss_fn in self.loss_functions.items():
                        loss_value = loss_fn(hand_pred, hand_gt, reduce=False)
                        all_hand_metrics[loss_name].append(loss_value.cpu())

                    for i in range(batch_size):
                        # Extract per-sample losses
                        sample_losses = {}
                        for loss_name, loss_tensor in all_hand_metrics.items():
                            latest_values = loss_tensor[-1]
                            sample_losses[loss_name] = latest_values[i].item()

                        current_predictions[i] = {**current_predictions[i], **sample_losses}

                self.predictions.extend(current_predictions)

            metrics_str = self._format_live_metrics(all_hand_metrics)
            progress.update(eval_task, advance=1, metrics=metrics_str)
            if (
                not use_rich
                and self.is_local_process_zero()
                and (step % 20 == 0 or step == len(dataloader) - 1)
            ):
                log.info(f"Eval {step + 1}/{len(dataloader)}  {metrics_str}".rstrip())

        progress.stop()

        # Aggregate metrics
        metrics = {}
        # Add hand metrics
        for metric_name, values in all_hand_metrics.items():
            metrics[f"{metric_key_prefix}_{metric_name}"] = np.mean(np.concatenate(values))
            metrics[f"{metric_key_prefix}_{metric_name}_std"] = np.std(np.concatenate(values))

        # Add inference time metrics
        if inference_times_per_sample:
            metrics[f"{metric_key_prefix}_inference_time_per_sample_mean"] = np.mean(
                inference_times_per_sample
            )
            metrics[f"{metric_key_prefix}_inference_time_per_sample_std"] = np.std(
                inference_times_per_sample
            )
            log.info(
                f"Average inference time per sample: {np.mean(inference_times_per_sample) * 1000:.2f}ms"
            )

        # Log metrics
        self.log(metrics)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )
