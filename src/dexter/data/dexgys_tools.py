"""Non-training DexGYS dataset variants.

These datasets are used only by utility/visualization/eval/benchmark scripts,
not by the active training path. They live here (rather than in
``dexgys.py``) but reuse the active ``DexGYSDataset`` via subclassing.
"""

import json
from pathlib import Path
from typing import Any, SupportsIndex

import h5py
import numpy as np
from torch.utils.data import Dataset

from dexter.data.dexgys import DexGYSDataset
from dexter.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class DexGYSHDF5Dataset(Dataset):
    """Legacy DexGYS dataset reading from HDF5 format.

    Used by utility scripts (e.g. compute_norm_stats).
    For training/evaluation, use DexGYSDataset instead.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        overfitting=False,
    ):
        """Initialize DexGYS HDF5 dataset.

        Args:
            data_path: Path to HDF5 dataset directory
            split: Dataset split ('train' or 'test')
            transform: Optional transform pipeline to apply to samples
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.overfitting = overfitting

        # Construct HDF5 file path
        if data_path:
            self.hdf5_path = Path(data_path) / f"dataset_{self.split}.h5"
        else:
            raise ValueError("data_path must be specified in data_config")

        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 dataset file not found: {self.hdf5_path}. "
                f"Please run scripts/convert_dexgys_to_hdf5.py first to convert the dataset."
            )

        # Don't open file here - open it in each worker process
        self.h5_file = None

        # Load metadata for efficient indexing
        self._load_metadata()

        log.info(
            f"Loaded DexGYS HDF5 dataset from {self.hdf5_path} | "
            f"split: {self.split} | "
            f"scenes: {self.num_scenes} | "
            f"samples: {self.num_samples}"
        )

    def _load_metadata(self):
        """Load metadata for efficient dataset indexing."""
        # Temporarily open file to load metadata
        with h5py.File(self.hdf5_path, "r") as h5_file:
            metadata_group = h5_file["metadata"]

            # Load scene information
            self.scene_names = [name.decode("utf-8") for name in metadata_group["scene_names"][:]]
            self.scene_query_counts = metadata_group["scene_query_counts"][:]

            # Load flat indices for O(1) sample access
            self.flat_indices = metadata_group["flat_indices"][:]

            # Store dataset statistics
            self.num_scenes = len(self.scene_names)
            self.num_samples = len(self.flat_indices)

            # Validate metadata consistency
            assert self.num_samples == metadata_group.attrs["num_samples"], (
                "Metadata inconsistency: sample count mismatch"
            )
            assert self.num_scenes == metadata_group.attrs["num_scenes"], (
                "Metadata inconsistency: scene count mismatch"
            )

    def _ensure_file_open(self):
        """Ensure HDF5 file is open for this worker process."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Dictionary containing the sample data in the expected format for the model.
        """
        idx = index.__index__()

        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")

        # Ensure file is open in this worker process
        self._ensure_file_open()

        # Get scene and query indices from flat index
        scene_idx, query_idx = self.flat_indices[idx]
        scene_name = self.scene_names[scene_idx]

        # Access data efficiently from HDF5
        scene_group = self.h5_file["scenes"][scene_name]
        query_group = scene_group["queries"][str(query_idx)]

        # Load data
        pointcloud = scene_group["pointcloud"][:]
        mask = query_group["mask"][:]
        grasp = query_group["grasp"][:]
        query_text = query_group["text"][()].decode("utf-8")

        if pointcloud.shape[-1] == 3:
            pointcloud = np.concatenate([pointcloud, np.ones_like(pointcloud) * 0.5], axis=-1)

        # Convert to the format expected by the model
        data = {
            "pointcloud": pointcloud.astype(np.float32),
            "actions": grasp.astype(np.float32),
            "prompt": query_text,
            "mask": mask.astype(np.float32),  # Include mask for potential future use
            "scene_name": scene_name,
        }

        # Apply transforms if provided
        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples if not self.overfitting else min(self.num_samples, 1)

    def __del__(self):
        """Clean up HDF5 file handle."""
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

    def close(self):
        """Explicitly close the HDF5 file."""
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

    def get_scene_info(self, scene_idx: int) -> dict[str, Any]:
        """Get information about a specific scene.

        Args:
            scene_idx: Index of the scene.

        Returns:
            Dictionary containing scene information.
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise IndexError(f"Scene index {scene_idx} out of range")

        scene_name = self.scene_names[scene_idx]
        query_count = self.scene_query_counts[scene_idx]

        return {
            "scene_name": scene_name,
            "query_count": query_count,
            "scene_idx": scene_idx,
        }

    def get_dataset_stats(self) -> dict[str, Any]:
        """Get overall dataset statistics.

        Returns:
            Dictionary containing dataset statistics.
        """
        return {
            "split": self.split,
            "num_scenes": self.num_scenes,
            "num_samples": self.num_samples,
            "avg_queries_per_scene": np.mean(self.scene_query_counts),
            "min_queries_per_scene": np.min(self.scene_query_counts),
            "max_queries_per_scene": np.max(self.scene_query_counts),
        }


class DexGYSVisualizationDataset(DexGYSDataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        overfitting=False,
    ):
        super().__init__(data_path, split, transform, overfitting)
        self.mesh_dir = self.data_path / "meshes"
        # read metadata
        self.object_id_to_mesh_path = {}

        for filename, dirname in [
            ("object_id.json", "OakInkObjectsV2"),
            ("virtual_object_id.json", "OakInkVirtualObjectsV2"),
        ]:
            with open(self.data_path / "meshes" / "metaV2" / filename, "r") as f:
                data = json.load(f)

            for key, value in data.items():
                mesh_dir = self.data_path / "meshes" / dirname / value["name"] / "align_ds"
                obj_files = list(mesh_dir.glob("*.obj"))
                ply_files = list(mesh_dir.glob("*.ply"))
                mesh_files = obj_files + ply_files

                mesh_path = mesh_files[0]
                self.object_id_to_mesh_path[key] = mesh_path

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        data = super().__getitem__(index)
        mesh_path = self.object_id_to_mesh_path[data["scene_name"]]
        data["mesh_path"] = mesh_path
        return data


class DexGYSPredictionDataset(DexGYSDataset):
    def __init__(self, data_path: str, prediction_path: str):
        super().__init__(data_path, split="test", transform=None, overfitting=False)
        self.data_path = Path(data_path)
        self.prediction_path = Path(prediction_path)

        self.mesh_dir = self.data_path / "meshes"
        # read metadata
        self.object_id_to_mesh_path = {}

        # load mesh paths
        for filename, dirname in [
            ("object_id.json", "OakInkObjectsV2"),
            ("virtual_object_id.json", "OakInkVirtualObjectsV2"),
        ]:
            with open(self.data_path / "meshes" / "metaV2" / filename, "r") as f:
                data = json.load(f)

            for key, value in data.items():
                mesh_dir = self.data_path / "meshes" / dirname / value["name"] / "align_ds"
                obj_files = list(mesh_dir.glob("*.obj"))
                ply_files = list(mesh_dir.glob("*.ply"))
                mesh_files = obj_files + ply_files

                mesh_path = mesh_files[0]
                self.object_id_to_mesh_path[key] = mesh_path
        # load predictions
        with open(self.prediction_path, "r") as f:
            self.predictions = json.load(f)

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        data = super().__getitem__(index)
        prediction = self.predictions[index]
        obj_id = prediction["obj_id"]

        mesh_path = self.object_id_to_mesh_path[obj_id]
        prompt = prediction["guidance"]
        pred_grasp = prediction["predictions"]
        gt_grasp = prediction["targets"]
        contact = prediction.get("contact", None)
        return_dict = {
            "mesh_path": mesh_path,
            "prompt": prompt,
            "pred_grasp": pred_grasp,
            "gt_grasp": gt_grasp,
            **data,
        }
        if contact is not None:
            return_dict["pred_contact"] = contact
        if prediction.get("sim_contact", None) is not None:
            return_dict["sim_contact"] = {
                k: np.array(v).mean(axis=0, keepdims=True).astype(np.float32)
                for k, v in prediction["sim_contact"].items()
            }
        return return_dict
