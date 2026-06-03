"""DexGYS Dataset for dexterous grasping with Shadow Hand."""

import json
from pathlib import Path
from typing import Any, Literal, SupportsIndex

import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

from dexter.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class DexonomyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        overfitting=False,
        query_type: Literal["low_level", "high_level", "template"] = "low_level",
        grasp_type: Literal["grasp", "pregrasp", "squeezegrasp"] = "grasp",
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.overfitting = overfitting
        self.query_type = query_type
        self.grasp_type = grasp_type

        split_file = self.data_path / "splits" / f"{self.split}.json"
        with open(split_file, "r") as f:
            files = json.load(f)
            files = natsorted(files)

        self.files = [self.data_path / file for file in files]

        log.info(
            f"Loaded Dexonomy dataset from {self.data_path} | "
            f"split: {self.split} | "
            f"samples: {len(self.files)} | "
            f"query_type: {self.query_type} | "
            f"grasp_type: {self.grasp_type}"
        )

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Dictionary containing the sample data in the expected format for the model.
        """
        idx = index.__index__()

        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range for dataset of size {self.__len__()}")

        instance_dir = self.files[idx]
        query_file = instance_dir / "grasps_queries.json"
        contact_file = instance_dir / "contact_points_info.json"
        pcd_file = instance_dir / "object_points.npy"
        contact_map_file = instance_dir / "contact_maps.npy"
        grasp_file = instance_dir / "grasps.npy"

        # Path structure: .../taxonomy/object_id/floating/scale_id
        scale_id = instance_dir.name  # e.g., "scale011"
        object_id = instance_dir.parent.parent.name  # e.g., "023d5f..."
        taxonomy = instance_dir.parent.parent.parent.name  # e.g., "1_Large_Diameter"
        taxonomy_label = " ".join(taxonomy.split("_")[1:])
        scene_name = f"{taxonomy}/{object_id}/{scale_id}"

        with open(query_file, "r") as f:
            query_data = json.load(f)[0]

        class_name = query_data.get("class_name", "")
        object_part = query_data.get("grasped_object_part", "")
        query_lowlevel = query_data.get("low_level_grasp_instruction", "")
        query_highlevel = query_data.get("high_level_grasp_instruction", "")
        query_templated = f"Grasp the object with the {taxonomy_label} grasp."

        if self.query_type == "template":
            prompt = query_templated
        elif self.query_type == "low_level":
            prompt = query_lowlevel
        elif self.query_type == "high_level":
            prompt = query_highlevel
        else:
            raise ValueError(f"Invalid query type: {self.query_type}")

        with open(contact_file, "r") as f:
            contact_data = json.load(f)[0]

        new_contact_dict = {}
        for k, v in contact_data.items():
            if len(v) == 0:
                continue
            new_contact_dict[k] = np.array(v).mean(axis=0, keepdims=True).astype(np.float32)

        pointcloud = np.load(pcd_file)
        mask = np.load(contact_map_file)[0]

        if pointcloud.shape[-1] == 3:
            pointcloud = np.concatenate([pointcloud, np.ones_like(pointcloud) * 0.5], axis=-1)

        grasp_data = np.load(grasp_file, allow_pickle=True).item()
        grasp = grasp_data.get(self.grasp_type + "_qpos", None)
        if grasp is None:
            grasp = grasp_data["grasp_qpos"]
        grasp = np.squeeze(grasp)

        # Convert to the format expected by the model
        data = {
            "pointcloud": pointcloud.astype(np.float32),
            "actions": grasp.astype(np.float32),
            "prompt": prompt,
            "prompt_lowlevel": query_lowlevel,
            "prompt_highlevel": query_highlevel,
            "prompt_templated": query_templated,
            "contact": new_contact_dict,
            "object_part": object_part,
            "class_name": class_name,
            "mask": mask.astype(np.float32),
            "scene_name": scene_name,
        }
        # Apply transforms if provided
        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files) if not self.overfitting else min(len(self.files), 2)


class DexonomyPredictionDataset(Dataset):
    def __init__(self, data_path: str, prediction_path: str, load_pcd: bool = False):
        self.data_path = Path(data_path)
        self.prediction_path = Path(prediction_path)
        self.load_pcd = load_pcd
        with open(self.prediction_path, "r") as f:
            self.predictions = json.load(f)

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        prediction = self.predictions[index]
        obj_id = prediction["obj_id"]
        prompt = prediction["guidance"]
        pred_grasp = prediction["predictions"]
        gt_grasp = prediction["targets"]
        pred_grasp = np.array(pred_grasp)
        gt_grasp = np.array(gt_grasp)

        if "obj_pc_path" in prediction:
            pcd_path = "/".join(prediction["obj_pc_path"].split("/")[-4:])
        else:
            parts = obj_id.split("/")
            parts = [*parts[:-1], "floating", parts[-1]]
            pcd_path = self.data_path / "/".join(parts) / "0_combined_points.npy"

        return_dict = {
            "pcd_path": pcd_path,
            "prompt": prompt,
            "pred_grasp": pred_grasp,
            "gt_grasp": gt_grasp,
        }
        if self.load_pcd:
            pcd = np.load(pcd_path)[:, :3]
            return_dict["pcd"] = pcd
        return return_dict


if __name__ == "__main__":
    """Quick smoke test: build the default Dexonomy pipeline and report token stats."""
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    from dexter.data.normalize import load as load_norm_stats
    from dexter.data.transforms import build_transforms
    from dexter.models.action_tokenizer import CONTACT_JOINT_NAMES, GraspTokenizerQwen3

    config = OmegaConf.load("configs/data/dexonomy.yaml")
    norm_stats = load_norm_stats(config.norm_stats_path)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    base_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    special_tokens = [f"<action_bin_{i}>" for i in range(256)]
    special_tokens += ["<|action_start|>", "<|action_end|>"]
    special_tokens += [f"<pos_bin_{i}>" for i in range(256)]
    special_tokens += ["<|pos_start|>", "<|pos_end|>"]
    special_tokens += [f"<{joint_name}>" for joint_name in CONTACT_JOINT_NAMES]
    special_tokens += ["<|joint_start|>", "<|joint_end|>"]
    base_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    action_tokenizer = GraspTokenizerQwen3(
        base_tokenizer,
        bins=256,
        min_action=-1.0,
        max_action=1.0,
        position_bins=256,
        min_position=-0.2,
        max_position=0.2,
    )

    transforms = build_transforms(config.transforms, norm_stats, base_tokenizer, action_tokenizer)
    dataset = DexonomyDataset(data_path="/root/data/dexonomy", split="train", transform=transforms)
    print(f"dataset length: {len(dataset)}")
    num_tokens = [dataset[i]["tokenized_prompt_mask"].sum() for i in range(min(1000, len(dataset)))]
    print(
        f"tokens — avg: {np.mean(num_tokens)}, max: {np.max(num_tokens)}, min: {np.min(num_tokens)}"
    )
