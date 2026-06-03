"""Normalization utilities for data transforms."""

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class NormStats:
    """Statistics for normalization.

    Attributes:
        mean: Mean values for z-score normalization
        std: Standard deviation values for z-score normalization
        q01: 1st percentile for quantile normalization (optional)
        q99: 99th percentile for quantile normalization (optional)
    """

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None
    max: np.ndarray | None = None
    min: np.ndarray | None = None


class RunningStats:
    """Compute running statistics of a batch of vectors.

    Supports both z-score statistics (mean/std) and quantile statistics (q01/q99)
    computed incrementally from batches of data.
    """

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            NormStats: A dataclass containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self._mean, std=stddev, q01=q01, q99=q99, max=self._max, min=self._min
        )

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize normalization statistics to a JSON string.

    Args:
        norm_stats: Dictionary mapping field names to NormStats

    Returns:
        JSON string representation
    """
    data = {}
    for key, stats in norm_stats.items():
        data[key] = {
            "mean": stats.mean.tolist(),
            "std": stats.std.tolist(),
            "q01": stats.q01.tolist() if stats.q01 is not None else None,
            "q99": stats.q99.tolist() if stats.q99 is not None else None,
            "max": stats.max.tolist() if stats.max is not None else None,
            "min": stats.min.tolist() if stats.min is not None else None,
        }
    return json.dumps({"norm_stats": data}, indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize normalization statistics from a JSON string.

    Args:
        data: JSON string

    Returns:
        Dictionary mapping field names to NormStats
    """
    parsed = json.loads(data)
    norm_stats = {}
    for key, stats_dict in parsed["norm_stats"].items():
        if key == "state":
            # skip state stats
            continue
        norm_stats[key] = NormStats(
            mean=np.array(stats_dict["mean"]).astype(np.float32),
            std=np.array(stats_dict["std"]).astype(np.float32),
            q01=(
                np.array(stats_dict["q01"]).astype(np.float32)
                if stats_dict["q01"] is not None
                else None
            ),
            q99=(
                np.array(stats_dict["q99"]).astype(np.float32)
                if stats_dict["q99"] is not None
                else None
            ),
        )
    return norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save normalization stats to a directory.

    Args:
        directory: Directory path to save to
        norm_stats: Dictionary mapping field names to NormStats
    """
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load normalization stats from a directory.

    Args:
        directory: Directory path to load from

    Returns:
        Dictionary mapping field names to NormStats
    """
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
