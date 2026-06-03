import numpy as np
import torch


def resample_pointcloud(
    pc: np.ndarray,
    target_size: int,
    seed: int | None = None,
) -> np.ndarray:
    """Resample point cloud to a fixed target size.

    If the point cloud has fewer points than target_size, points are
    randomly duplicated (sampled with replacement) to reach the target.
    If it has more points, random subsampling is performed.

    This approach ensures all points in the output are valid real points,
    which is important for encoders that use FPS, KNN, or voxelization.
    Padding with zeros would corrupt features in these operations.

    Args:
        pc: Point cloud array of shape (N, C) where C is typically 6 (XYZ + RGB)
        target_size: Target number of points
        seed: Optional random seed for reproducibility

    Returns:
        Resampled point cloud of shape (target_size, C)
    """
    n_points = len(pc)

    if n_points == target_size:
        return pc

    rng = np.random.default_rng(seed)

    if n_points < target_size:
        # Upsample: duplicate points randomly
        # First keep all original points, then sample additional points with replacement
        extra_needed = target_size - n_points
        extra_indices = rng.choice(n_points, size=extra_needed, replace=True)
        indices = np.concatenate([np.arange(n_points), extra_indices])
    else:
        # Downsample: randomly select points without replacement
        indices = rng.choice(n_points, size=target_size, replace=False)

    return pc[indices]


class Collator:
    """
    Collator for Dexter model.

    Handles:
    - Batching pre-transformed samples
    - Renaming keys to model-expected format
    - Resampling variable-size point clouds to a fixed size for batching

    Note: Tokenization, normalization, and padding are handled by transforms
    in the dataset, not by the collator.

    Args:
        max_points: Target size for point cloud resampling. If None, point clouds
            must have the same size. If specified, all point clouds will be
            resampled to this size, enabling batched inference with variable-size
            inputs (e.g., from partial observation).
    """

    def __init__(self, max_points: int | None = None):
        """Initialize the collator.

        Args:
            max_points: Target size for point cloud resampling. If None (default),
                point clouds must have the same size for batching. Set this when
                using partial observation to enable batch_size > 1.
        """
        self.max_points = max_points

    def __call__(self, batch):
        """
        Collate pre-transformed samples into a batch.

        Tokenization, normalization, and padding are handled by the transform
        pipeline, so this just stacks fields, resamples variable-size point
        clouds to a common size, and drops truncated samples. Each item is
        expected to carry:
        - pointcloud: (N, 6) point cloud
        - tokenized_prompt: tokenized text
        - tokenized_prompt_mask: attention mask for text
        - actions: (action_horizon, action_dim) actions
        - mask: (N,) segmentation mask (optional)
        """
        return_batch = {}

        # Check if point clouds have variable sizes
        pc_sizes = [len(item["pointcloud"]) for item in batch]
        need_resampling = len(set(pc_sizes)) > 1

        if need_resampling and self.max_points is None:
            # Auto-determine max_points from the batch
            self.max_points = max(pc_sizes)

        for k, v in batch[0].items():
            if k == "pointcloud" and self.max_points is not None:
                # Resample point clouds to fixed size for batching
                # Handle both numpy arrays and tensors
                resampled_pcs = []
                for item in batch:
                    pc = item[k]
                    if isinstance(pc, torch.Tensor):
                        pc = pc.numpy()
                    resampled_pcs.append(resample_pointcloud(pc, self.max_points))
                return_batch[k] = torch.stack([torch.from_numpy(pc) for pc in resampled_pcs])
            elif isinstance(v, np.ndarray):
                return_batch[k] = torch.stack([torch.from_numpy(item[k]) for item in batch])
            elif isinstance(v, torch.Tensor):
                return_batch[k] = torch.stack([item[k] for item in batch])
            else:
                return_batch[k] = [item[k] for item in batch]

        # Handle original action dimension
        original_action_dim = return_batch.pop("original_action_dim", None)
        if original_action_dim is not None:
            original_action_dim = original_action_dim[0]

        # Filter out truncated samples
        valid_sample_mask = return_batch["tokenized_prompt_mask"][:, -1] == 0
        for k in return_batch.keys():
            if isinstance(return_batch[k], torch.Tensor):
                return_batch[k] = return_batch[k][valid_sample_mask]
            else:
                return_batch[k] = [
                    item
                    for batch_idx, item in enumerate(return_batch[k])
                    if valid_sample_mask[batch_idx]
                ]

        return {
            **return_batch,
            "original_action_dim": original_action_dim,
        }
