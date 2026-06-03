"""DexGYS Dataset for dexterous grasping with Shadow Hand."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np
from torch.utils.data import Dataset

from dexter.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class DexGYSDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        overfitting=False,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.overfitting = overfitting

        self.flatten_indices = self.load_metadata()

        log.info(
            f"Loaded DexGYS CoT dataset from {self.data_path} | "
            f"split: {self.split} | "
            f"num_scenes: {len(set([scene_name for scene_name, _ in self.flatten_indices]))} | "
            f"samples: {len(self.flatten_indices)}"
        )

    def load_metadata(self):
        with open(self.data_path / f"{self.split}.txt") as f:
            lines = [line.strip() for line in f.readlines()]
            scene_name_to_query_indices = defaultdict(list)
            for line in lines:
                parts = line.split(":")
                scene_name = parts[0]
                scene_dir = self.data_path / "data" / scene_name
                if not scene_dir.exists():
                    continue
                if not (scene_dir / "contact_points_info.json").exists():
                    continue

                if len(parts) > 1:
                    query_index = int(parts[1])
                    scene_name_to_query_indices[scene_name].append(query_index)
                else:
                    with open(scene_dir / "grasps.json") as gf:
                        num_grasps = len(json.load(gf))
                    scene_name_to_query_indices[scene_name] = list(range(num_grasps))

        flatten_indices = []
        for scene_name, query_indices in scene_name_to_query_indices.items():
            for query_index in query_indices:
                flatten_indices.append((scene_name, query_index))
        return flatten_indices

    def __len__(self):
        num_samples = len(self.flatten_indices)
        return num_samples if not self.overfitting else min(num_samples, 1)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        idx = index.__index__()
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range for dataset of size {self.__len__()}")

        scene_name, query_index = self.flatten_indices[idx]

        scene_dir = self.data_path / "data" / scene_name

        with open(scene_dir / "grasps.json") as f:
            grasps = json.load(f)
        grasp = grasps[query_index]

        with open(scene_dir / "queries.json") as f:
            query_data = json.load(f)[0]

        class_name = query_data["class_name"]
        query = query_data["queries"][query_index]

        with open(scene_dir / "contact_points_info.json") as f:
            all_contact_data = json.load(f)
        contact_data = all_contact_data[query_index]

        for k, v in contact_data.items():
            contact_data[k] = np.array(v).mean(axis=0, keepdims=True).astype(np.float32)

        xyzc = np.load(scene_dir / "xyzc.npy")
        pointcloud = xyzc[:, :3]
        mask = xyzc[:, 3 + query_index]

        if pointcloud.shape[-1] == 3:
            pointcloud = np.concatenate([pointcloud, np.ones_like(pointcloud) * 0.5], axis=-1)

        data = {
            "pointcloud": pointcloud.astype(np.float32),
            "actions": np.array(grasp).astype(np.float32),
            "prompt": query,
            "mask": mask.astype(np.float32),
            "scene_name": scene_name,
            "class_name": class_name,
            "contact": contact_data,
        }
        if self.transform is not None:
            data = self.transform(data)
        return data


# Backward compatibility alias
DexGYSCoTDataset = DexGYSDataset


if __name__ == "__main__":
    """Quick smoke test: build the default DexGYS pipeline and report token stats."""
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    from dexter.data.normalize import load as load_norm_stats
    from dexter.data.transforms import build_transforms
    from dexter.models.action_tokenizer import CONTACT_JOINT_NAMES, GraspTokenizerQwen3

    config = OmegaConf.load("configs/data/dexgys.yaml")
    norm_stats = load_norm_stats(config.norm_stats_path)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    base_tokenizer = processor.tokenizer

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
    dataset = DexGYSDataset(data_path="/root/data/dexgys", split="test", transform=transforms)
    print(f"dataset length: {len(dataset)}")
    num_tokens = [dataset[i]["tokenized_prompt_mask"].sum() for i in range(min(1000, len(dataset)))]
    print(
        f"tokens — avg: {np.mean(num_tokens)}, max: {np.max(num_tokens)}, min: {np.min(num_tokens)}"
    )
