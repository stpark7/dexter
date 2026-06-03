"""Shared viser visualization infrastructure for the ``scripts/visualize_*`` tools.

This module centralizes the pieces every visualization script used to duplicate:
color/setting constants, point-cloud coloring, a backend-agnostic Shadow Hand
forward-kinematics wrapper (:class:`UnifiedHandModel`), and a :class:`BaseVisualization`
base class that owns the viser server lifecycle.
"""

import time
from contextlib import suppress
from typing import Any

import numpy as np
import torch
import transforms3d
import viser

from dexter.utils.shadowhand import ShadowHandModel
from dexter.utils.shadowhand_mujoco import RobotKinematics

# Default asset locations
SHADOWHAND_ASSET_DIR = "./assets/shadowhand"
MUJOCO_XML_PATH = "./assets/shadowhand_mujoco/right_hand.xml"

# Wrist offset (meters) applied along the hand's local +Z when converting a
# DexGYS grasp into the MuJoCo root pose.
MUJOCO_WRIST_OFFSET = np.array([0.0, 0.0, 0.034])


class Colors:
    """RGB color constants (values in [0, 1])."""

    HAND = np.array([70, 130, 227]) / 255.0  # blue (predicted / default hand)
    HAND_GT = (1.0, 0.5, 0.2)  # orange (ground-truth hand)
    GRAY = (0.5, 0.5, 0.5)
    MASK_FOREGROUND = (1.0, 0.3, 0.3)  # red (masked points)
    MASK_BACKGROUND = (0.5, 0.5, 0.5)  # gray (background points)
    CONTACT_POINT = np.array([234, 196, 81]) / 255.0  # gold
    PRED_CONTACT_POINT = (1.0, 0.0, 0.0)  # red


class Settings:
    """Default visualization settings."""

    HAND_OPACITY_DEFAULT = 0.7
    CONTACT_POINT_RADIUS = 0.008
    # Partial observation defaults (match scripts/test.py defaults)
    PARTIAL_OBS_MODE = "ego_thirdperson"
    PARTIAL_OBS_CAMERA_RADIUS = 0.3
    PARTIAL_OBS_SEED = 42
    PARTIAL_OBS_DEPTH_NOISE = 2.0  # mm
    PARTIAL_OBS_LATERAL_NOISE = 1.0  # mm
    PARTIAL_OBS_OUTLIER_RATIO = 1.0  # %


def parse_grasp_28d(grasp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a grasp vector into MuJoCo (joint_qpos, root_pose).

    A 28-D DexGYS grasp is laid out as ``[translation (3), axis-angle (3),
    joints (22)]``. It is converted to ``root_pose = [x, y, z, qw, qx, qy, qz]``
    (with the wrist offset applied) plus the trailing joint angles. Inputs that
    are already in ``[pose (7), joints]`` layout are split as-is.

    Args:
        grasp: Grasp array, last dim 28 (DexGYS) or already pose+joints.

    Returns:
        Tuple of ``(joint_qpos, root_pose)``.
    """
    if grasp.shape[-1] == 28:
        t, r = grasp[..., :3], grasp[..., 3:6]
        rot = transforms3d.axangles.axangle2mat(r, np.linalg.norm(r))
        quat = transforms3d.quaternions.mat2quat(rot)
        offset_world = rot @ MUJOCO_WRIST_OFFSET
        new_grasp = np.concatenate([t - offset_world, quat, grasp[..., 6:]])
        return new_grasp[..., 7:], new_grasp[..., :7]

    return grasp[..., 7:], grasp[..., :7]


def pointcloud_colors(
    pointcloud: np.ndarray,
    mask: np.ndarray | None = None,
    show_mask: bool = False,
    show_rgb: bool = True,
) -> np.ndarray:
    """Determine per-point colors for a point cloud based on display settings.

    Args:
        pointcloud: Point cloud of shape ``(N, D)`` with ``D >= 3``.
        mask: Optional segmentation mask of shape ``(N,)``.
        show_mask: Color by segmentation mask (foreground red / background gray).
        show_rgb: Use the cloud's RGB channels (``[:, 3:6]``) when available.

    Returns:
        Color array of shape ``(N, 3)``.
    """
    n = len(pointcloud)
    if show_mask and mask is not None:
        colors = np.zeros((n, 3))
        colors[mask > 0.5] = Colors.MASK_FOREGROUND
        colors[mask <= 0.5] = Colors.MASK_BACKGROUND
        return colors
    if show_rgb and pointcloud.shape[1] >= 6:
        return pointcloud[:, 3:6]
    return np.full((n, 3), Colors.GRAY)


class UnifiedHandModel:
    """Lazy Shadow Hand forward kinematics over either backend.

    Wraps the analytic :class:`ShadowHandModel` or the MuJoCo-based
    :class:`RobotKinematics`, exposing a single :meth:`compute_meshes` API that
    returns batched vertices ``[B, V, 3]`` and unbatched faces ``[F, 3]`` for both.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_mujoco: bool = False,
        shadowhand_dir: str = SHADOWHAND_ASSET_DIR,
        mujoco_xml: str = MUJOCO_XML_PATH,
    ):
        self.device = device
        self.use_mujoco = use_mujoco
        self.shadowhand_dir = shadowhand_dir
        self.mujoco_xml = mujoco_xml
        self._model: ShadowHandModel | RobotKinematics | None = None

    @property
    def model(self) -> ShadowHandModel | RobotKinematics:
        """Lazily construct the underlying hand model."""
        if self._model is None:
            if self.use_mujoco:
                self._model = RobotKinematics(xml_path=self.mujoco_xml)
            else:
                self._model = ShadowHandModel(
                    base_dir=self.shadowhand_dir, device=self.device, vis=True
                )
        return self._model

    def compute_meshes(
        self,
        grasps: Any,
        parse_28d: bool = False,
        with_penetration: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute hand mesh(es) for one or more grasps.

        Args:
            grasps: Array/tensor/list of shape ``[D]`` or ``[B, D]``.
            parse_28d: Convert 28-D DexGYS grasps via :func:`parse_grasp_28d`
                before MuJoCo FK. Ignored for the analytic backend (which
                consumes 28-D grasps directly).
            with_penetration: Request penetration depth (analytic backend only).

        Returns:
            Dict with ``vertices`` ``[B, V, 3]`` and ``faces`` ``[F, 3]`` numpy
            arrays, plus ``penetration`` ``[B]`` when available.
        """
        if self.use_mujoco:
            return self._compute_meshes_mujoco(grasps, parse_28d=parse_28d)
        return self._compute_meshes_analytic(grasps, with_penetration=with_penetration)

    def _compute_meshes_analytic(
        self, grasps: Any, with_penetration: bool
    ) -> dict[str, np.ndarray]:
        pose = torch.as_tensor(np.asarray(grasps), dtype=torch.float32).to(self.device)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)

        with torch.no_grad():
            out = self.model(pose, with_meshes=True, with_penetration=with_penetration)

        result = {
            "vertices": out["vertices"].detach().cpu().numpy().astype(np.float32),
            "faces": out["faces"].detach().cpu().numpy(),
        }
        if with_penetration and out.get("penetration") is not None:
            result["penetration"] = out["penetration"].detach().cpu().numpy().astype(np.float32)
        return result

    def _compute_meshes_mujoco(self, grasps: Any, parse_28d: bool) -> dict[str, np.ndarray]:
        grasps = np.asarray(grasps, dtype=np.float32)
        if grasps.ndim == 1:
            grasps = grasps[None]

        vertices: list[np.ndarray] = []
        faces: np.ndarray | None = None
        for grasp in grasps:
            if parse_28d:
                qpos, pose = parse_grasp_28d(grasp)
            else:
                qpos, pose = grasp[..., 7:], grasp[..., :7]
            self.model.forward_kinematics(qpos)
            mesh = self.model.get_posed_meshes(pose)
            vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
            faces = np.asarray(mesh.faces)

        return {"vertices": np.stack(vertices), "faces": faces}


class BaseVisualization:
    """Base class owning the viser server and common rendering helpers."""

    def __init__(self, port: int, device: str, use_mujoco: bool = False):
        self.port = port
        self.device = device
        self.use_mujoco = use_mujoco
        self.hand = UnifiedHandModel(device=device, use_mujoco=use_mujoco)
        self.server: viser.ViserServer | None = None

    def _create_server(self, name: str) -> None:
        """Create the viser server."""
        self.server = viser.ViserServer(port=self.port, name=name)

    def _clear_scene(self, scene_names: list[str]) -> None:
        """Remove the named objects (and their children) from the scene."""
        for name in scene_names:
            with suppress(Exception):
                self.server.scene.remove_by_name(name)

    def _render_hand_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[float, float, float],
        opacity: float,
        offset: np.ndarray | None = None,
    ) -> None:
        """Render a hand mesh, optionally translated by ``offset``."""
        verts = vertices.copy()
        if offset is not None:
            verts += offset

        self.server.scene.add_mesh_simple(
            name,
            vertices=verts,
            faces=faces,
            color=color,
            opacity=opacity,
            wireframe=False,
        )

    @staticmethod
    def _update_markdown(markdown_widget, text: str) -> None:
        """Update a viser markdown widget's text in place."""
        markdown_widget._markdown = text  # noqa: SLF001

    def run_server_loop(self) -> None:
        """Block, serving the visualization until interrupted."""
        print(f"Visualization ready! Open http://localhost:{self.port} in your browser")
        print("Press Ctrl+C to stop the server")

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down server...")
            if self.server:
                self.server.stop()
