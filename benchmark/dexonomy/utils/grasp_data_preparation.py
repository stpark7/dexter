"""
Grasp data preparation utilities for DexGraspBench.
Converts prediction JSON into standardized grasp_data dicts for Dexonomy evaluation.
"""

import os

import numpy as np

from .file_util import load_json

_DEXTER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_DGBENCH_OBJECT_ROOT = os.path.join(_DEXTER_ROOT, "assets", "object")


def _remap_obj_path(obj_path: str) -> str:
    """Remap obj_path to the local copy under assets/object/."""
    marker = "assets/object/"
    idx = obj_path.find(marker)
    rel = obj_path[idx + len(marker) :]
    return os.path.join(_DGBENCH_OBJECT_ROOT, rel)


def prepare_prediction_grasp_data(prediction: dict, configs) -> dict:
    """
    Prepare grasp_data dict from prediction JSON.

    Args:
        prediction: Prediction dict from JSON with keys:
            - obj_id: Object identifier
            - predictions: Predicted grasp pose
            - guidance: Optional text guidance

    Returns:
        grasp_data dict with all necessary fields for BaseEval, or None if invalid
    """
    grasp_data = {}

    grasp_qpos = np.array(prediction["predictions"])
    grasp_data["grasp_qpos"] = grasp_qpos
    base_dir = os.path.dirname(prediction["obj_id"])
    sub_dir = os.path.basename(prediction["obj_id"])
    grasp_path = os.path.join(base_dir, "floating", sub_dir)
    grasp_metadata = np.load(
        os.path.join(configs.task.data_path, grasp_path, "grasps.npy"), allow_pickle=True
    ).item()
    grasp_data["obj_scale"] = grasp_metadata["obj_scale"]
    grasp_data["obj_pose"] = grasp_metadata["obj_pose"]
    grasp_data["obj_path"] = _remap_obj_path(grasp_metadata["obj_path"])

    grasp_data["obj_id"] = os.path.basename(grasp_data["obj_path"])
    grasp_data["obj_info"] = load_json(os.path.join(grasp_data["obj_path"], "info/simplified.json"))

    # Calculate pregrasp_qpos and squeeze_qpos
    grasp_pose = grasp_data["grasp_qpos"][:7]  # translation (3) + quaternion (4)
    grasp_joints = grasp_data["grasp_qpos"][7:]  # joint angles (22)

    pregrasp_joints = grasp_joints * 1.0
    squeeze_joints = grasp_joints * 1.2

    grasp_data["pregrasp_qpos"] = np.concatenate([grasp_pose, pregrasp_joints])
    grasp_data["squeeze_qpos"] = np.concatenate([grasp_pose, squeeze_joints])
    grasp_data["approach_qpos"] = None

    return grasp_data
