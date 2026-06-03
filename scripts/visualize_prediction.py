"""Interactive visualization comparing predicted vs. ground-truth grasps.

Renders the predicted hand (blue) and the ground-truth hand (orange) over the
object, with optional contact points, contact-joint IoU, and an optional
partial RGB-D observation simulation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from dexter.data.dexgys_tools import DexGYSPredictionDataset
from dexter.data.dexonomy import DexonomyPredictionDataset
from dexter.data.rgbd_simulation import simulate_partial_rgbd_observation
from dexter.utils.viz import (
    BaseVisualization,
    Colors,
    Settings,
    pointcloud_colors,
)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class VisualizationData:
    """Data for single sample visualization."""

    pred_grasp: np.ndarray
    gt_grasp: np.ndarray
    prompt: str
    obj_vertices: np.ndarray | None = None
    obj_faces: np.ndarray | None = None
    point_clouds: np.ndarray | None = None
    pred_hand_verts: np.ndarray | None = None
    pred_hand_faces: np.ndarray | None = None
    pred_penetration: float | None = None
    gt_hand_verts: np.ndarray | None = None
    gt_hand_faces: np.ndarray | None = None
    gt_penetration: float | None = None
    contact: dict | None = None
    pred_contact: dict | None = None
    sim_contact: dict | None = None


# ============================================================================
# Mesh Loading
# ============================================================================
def load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh vertices and faces from an OBJ or PLY file."""
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        return vertices, faces
    except Exception as e:
        print(f"Warning: Could not load mesh from {mesh_path}: {e}")
        # Return dummy mesh
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 0, 0]], dtype=np.int32)
        return vertices, faces


def load_pcd(pcd_path: Path) -> np.ndarray:
    """Load point cloud XYZ from a PCD/NPY file."""
    pcd = np.load(pcd_path)
    return pcd[:, :3]


# ============================================================================
# Prediction Visualization
# ============================================================================
class GraspVisualization(BaseVisualization):
    """Interactive visualization for grasp predictions."""

    def __init__(self, dataset: DexGYSPredictionDataset, port: int = 8080, device: str = "cuda"):
        use_mujoco = isinstance(dataset, DexonomyPredictionDataset)
        super().__init__(port, device, use_mujoco)
        self.dataset = dataset
        self.exp_name = f"Predictions ({len(dataset)} samples)"

        self.current_data: VisualizationData | None = None
        self._partial_obs_stats: dict | None = None
        self._setup_gui()

    def _setup_gui(self):
        """Setup GUI elements."""
        self._create_server(self.exp_name)
        self.server.gui.add_markdown(f"**{self.exp_name}**")

        # Navigation buttons
        with self.server.gui.add_folder("Navigation"):
            self.prev_button = self.server.gui.add_button("← Previous")
            self.next_button = self.server.gui.add_button("Next →")
            self.sample_slider = self.server.gui.add_slider(
                "Sample Index", min=0, max=len(self.dataset) - 1, step=1, initial_value=0
            )

        # Display controls
        with self.server.gui.add_folder("Display Options"):
            self.show_pred = self.server.gui.add_checkbox(
                "Show Predicted (Blue)", initial_value=True
            )
            self.show_gt = self.server.gui.add_checkbox(
                "Show Ground Truth (Orange)", initial_value=True
            )
            self.show_contact_points = self.server.gui.add_checkbox(
                "Show Contact Points", initial_value=True
            )
            self.hand_opacity = self.server.gui.add_slider(
                "Hand Opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=Settings.HAND_OPACITY_DEFAULT,
            )
            self.offset_x = self.server.gui.add_slider(
                "Offset X", min=-1.0, max=1.0, step=0.05, initial_value=0.0
            )
            self.offset_y = self.server.gui.add_slider(
                "Offset Y", min=-1.0, max=1.0, step=0.05, initial_value=0.0
            )

        # Partial observation controls
        with self.server.gui.add_folder("Partial Observation"):
            self.enable_partial_obs = self.server.gui.add_checkbox(
                "Enable Partial Obs", initial_value=False
            )
            self.partial_obs_mode = self.server.gui.add_dropdown(
                "Camera Mode",
                options=["ego_thirdperson", "hemisphere"],
                initial_value=Settings.PARTIAL_OBS_MODE,
            )
            self.partial_obs_camera_radius = self.server.gui.add_slider(
                "Camera Radius",
                min=0.1,
                max=1.0,
                step=0.05,
                initial_value=Settings.PARTIAL_OBS_CAMERA_RADIUS,
            )
            self.enable_noise = self.server.gui.add_checkbox(
                "Add Sensor Noise", initial_value=False
            )
            self.depth_noise = self.server.gui.add_slider(
                "Depth Noise (mm)",
                min=0.0,
                max=10.0,
                step=0.5,
                initial_value=Settings.PARTIAL_OBS_DEPTH_NOISE,
            )
            self.lateral_noise = self.server.gui.add_slider(
                "Lateral Noise (mm)",
                min=0.0,
                max=5.0,
                step=0.5,
                initial_value=Settings.PARTIAL_OBS_LATERAL_NOISE,
            )
            self.outlier_ratio = self.server.gui.add_slider(
                "Outlier Ratio (%)",
                min=0.0,
                max=5.0,
                step=0.5,
                initial_value=Settings.PARTIAL_OBS_OUTLIER_RATIO,
            )
            self.partial_obs_info = self.server.gui.add_markdown("**Partial Obs:** Disabled")

        # Info display
        self.prompt_text = self.server.gui.add_markdown("")
        self.contact_iou_text = self.server.gui.add_markdown("**Contact Joint IoU:** N/A")

        self._register_callbacks()

    def _register_callbacks(self):
        """Register GUI callbacks."""
        # Button callbacks
        self.prev_button.on_click(lambda _: self._on_prev())
        self.next_button.on_click(lambda _: self._on_next())

        # Widget update callbacks (display only, no reload)
        for widget in [
            self.sample_slider,
            self.show_pred,
            self.show_gt,
            self.show_contact_points,
            self.hand_opacity,
            self.offset_x,
            self.offset_y,
        ]:
            widget.on_update(lambda _: self._update_visualization())

        # Partial observation callbacks (require data reload)
        for widget in [
            self.enable_partial_obs,
            self.partial_obs_mode,
            self.partial_obs_camera_radius,
            self.enable_noise,
            self.depth_noise,
            self.lateral_noise,
            self.outlier_ratio,
        ]:
            widget.on_update(lambda _: self._on_partial_obs_changed())

    def _on_prev(self):
        """Handle previous button click."""
        if self.sample_slider.value > 0:
            self.sample_slider.value -= 1

    def _on_next(self):
        """Handle next button click."""
        if self.sample_slider.value < len(self.dataset) - 1:
            self.sample_slider.value += 1

    def _on_partial_obs_changed(self):
        """Handle partial observation settings change (requires data reload)."""
        # Force reload by clearing last sample index
        self._last_sample_idx = None
        self._update_visualization()

    def _load_sample_data(self, sample_idx: int):
        """Load and preprocess data for a sample."""
        print(f"Loading sample {sample_idx}...")

        # Get sample from dataset
        sample = self.dataset[sample_idx]
        mesh_path = sample.get("mesh_path", None)
        pcd_path = None
        if mesh_path is None:
            pcd_path = sample["pcd_path"]

        prompt = sample["prompt"]
        pred_grasp = np.array(sample["pred_grasp"], dtype=np.float32)
        gt_grasp = np.array(sample["gt_grasp"], dtype=np.float32)

        # Load object mesh
        if mesh_path is not None:
            obj_vertices, obj_faces = load_mesh(mesh_path)
        else:
            obj_vertices = load_pcd(pcd_path)
            obj_faces = None

        # Apply partial observation simulation if enabled
        partial_obs_stats = None
        if self.enable_partial_obs.value:
            # Create point cloud with dummy RGB if needed (HPR only uses XYZ)
            if obj_vertices.shape[1] == 3:
                pointcloud = np.hstack([obj_vertices, np.ones((len(obj_vertices), 3)) * 0.5])
            else:
                pointcloud = obj_vertices

            partial_pc, visibility_mask, partial_obs_stats = simulate_partial_rgbd_observation(
                pointcloud=pointcloud,
                grasp_pose=gt_grasp,
                num_views=3,  # Used for hemisphere mode
                camera_radius=self.partial_obs_camera_radius.value,
                seed=Settings.PARTIAL_OBS_SEED,
                camera_mode=self.partial_obs_mode.value,
                add_noise=self.enable_noise.value,
                depth_noise_std=self.depth_noise.value / 1000.0,  # mm to meters
                lateral_noise_std=self.lateral_noise.value / 1000.0,  # mm to meters
                outlier_ratio=self.outlier_ratio.value / 100.0,  # % to ratio
            )

            # Update vertices with partial observation
            obj_vertices = partial_pc[:, :3].astype(np.float32)
            obj_faces = None  # Partial observation produces point cloud only

            print(
                f"Partial observation: {partial_obs_stats['original_points']} -> "
                f"{partial_obs_stats['visible_points']} points "
                f"({partial_obs_stats['visibility_ratio']:.1%} visible)"
            )

        # Store stats for display
        self._partial_obs_stats = partial_obs_stats

        print(f"Computing hand meshes for sample {sample_idx}...")
        pred_hand_out = self.hand.compute_meshes(pred_grasp, with_penetration=True)
        gt_hand_out = self.hand.compute_meshes(gt_grasp, with_penetration=True)

        self.current_data = VisualizationData(
            obj_vertices=obj_vertices,
            obj_faces=obj_faces,
            pred_grasp=pred_grasp,
            gt_grasp=gt_grasp,
            prompt=prompt,
            pred_hand_verts=pred_hand_out["vertices"][0],
            pred_hand_faces=pred_hand_out["faces"],
            pred_penetration=pred_hand_out.get("penetration", [None])[0],
            gt_hand_verts=gt_hand_out["vertices"][0],
            gt_hand_faces=gt_hand_out["faces"],
            gt_penetration=gt_hand_out.get("penetration", [None])[0],
            contact=sample.get("contact", None),
            pred_contact=sample.get("pred_contact", None),
            sim_contact=sample.get("sim_contact", None),
        )

        print(
            f"Object mesh: {obj_vertices.shape[0]} vertices, "
            f"{obj_faces.shape[0] if obj_faces is not None else 'None'} faces"
        )
        print(f"Prompt: {prompt}")

    def _update_visualization(self):
        """Update the visualization based on current settings."""
        sample_idx = self.sample_slider.value

        # Load sample if not already loaded or if index changed
        if self.current_data is None or sample_idx != getattr(self, "_last_sample_idx", None):
            self._load_sample_data(sample_idx)
            self._last_sample_idx = sample_idx

        self._clear_scene(
            [
                "/object",
                "/pred_hand",
                "/gt_hand",
                "/contact_points_gt",
                "/contact_points_pred",
                "/contact_points_sim",
            ]
        )
        self._update_prompt_display()
        self._update_partial_obs_info()
        self._render_object()
        self._render_hands()

        # Compute and display contact joint IoU
        self._update_contact_iou_display()

        if self.show_contact_points.value:
            self._render_contact_points(self.current_data.contact, postfix="gt")
            self._render_contact_points(self.current_data.pred_contact, postfix="pred")
            self._render_contact_points(self.current_data.sim_contact, postfix="sim")

    def _update_prompt_display(self):
        """Update the prompt text display."""
        self._update_markdown(self.prompt_text, f"**Prompt:** {self.current_data.prompt}")

    def _update_partial_obs_info(self):
        """Update the partial observation info display."""
        if not self.enable_partial_obs.value:
            self._update_markdown(self.partial_obs_info, "**Partial Obs:** Disabled")
            return

        stats = self._partial_obs_stats
        if stats is None:
            self._update_markdown(self.partial_obs_info, "**Partial Obs:** No stats")
            return

        noise_str = (
            f", noise={stats['depth_noise_std'] * 1000:.1f}mm" if stats.get("noise_applied") else ""
        )
        info_text = (
            f"**Partial Obs:** {stats['original_points']} → {stats['visible_points']} pts "
            f"({stats['visibility_ratio']:.1%}){noise_str}"
        )
        self._update_markdown(self.partial_obs_info, info_text)

    def _update_contact_iou_display(self):
        """Compute and display the IoU of contact joint names."""
        if (
            not self.current_data.contact
            or not self.current_data.pred_contact
        ):
            self._update_markdown(self.contact_iou_text, "**Contact Joint IoU:** N/A")
            return

        # Get sets of joint names
        gt_joints = set(self.current_data.contact.keys())
        pred_joints = set(self.current_data.pred_contact.keys())

        # Compute intersection and union
        intersection = gt_joints & pred_joints
        union = gt_joints | pred_joints
        iou = len(intersection) / len(union) if len(union) > 0 else 0.0

        iou_text = (
            f"**Contact Joint IoU:** {iou:.3f}\n"
            f"- GT Joints: {len(gt_joints)}\n"
            f"- Pred Joints: {len(pred_joints)}\n"
            f"- Intersection: {len(intersection)}\n"
            f"- Union: {len(union)}"
        )
        self._update_markdown(self.contact_iou_text, iou_text)

    def _render_object(self):
        """Render the object mesh or point cloud."""
        if self.current_data.obj_faces is not None:
            self.server.scene.add_mesh_simple(
                "/object",
                vertices=self.current_data.obj_vertices,
                faces=self.current_data.obj_faces,
                color=Colors.GRAY,
                opacity=1.0,
                wireframe=False,
            )
        elif self.current_data.obj_vertices is not None:
            self.server.scene.add_point_cloud(
                "/object",
                points=self.current_data.obj_vertices,
                colors=pointcloud_colors(self.current_data.obj_vertices, show_rgb=False),
                point_size=0.001,
            )

    def _render_hands(self):
        """Render predicted and ground truth hand meshes."""
        offset = np.array([self.offset_x.value, self.offset_y.value, 0.0], dtype=np.float32)

        # Predicted hand
        if self.show_pred.value and self.current_data.pred_hand_verts is not None:
            self._render_hand_mesh(
                "/pred_hand",
                self.current_data.pred_hand_verts,
                self.current_data.pred_hand_faces,
                Colors.HAND,
                self.hand_opacity.value,
            )

        # Ground truth hand
        if self.show_gt.value and self.current_data.gt_hand_verts is not None:
            self._render_hand_mesh(
                "/gt_hand",
                self.current_data.gt_hand_verts,
                self.current_data.gt_hand_faces,
                Colors.HAND_GT,
                self.hand_opacity.value,
                offset=offset,
            )

    def _render_contact_points(self, contact_data: dict, postfix: str = ""):
        """Render contact points with joint name labels if available."""
        if not contact_data:
            return

        for joint_name in sorted(contact_data.keys()):
            positions = np.array(contact_data[joint_name]).reshape(-1)

            self.server.scene.add_icosphere(
                f"/contact_points_{postfix}/sphere/{joint_name}",
                radius=Settings.CONTACT_POINT_RADIUS,
                color=Colors.PRED_CONTACT_POINT
                if postfix == "pred"
                else Colors.CONTACT_POINT,
                position=positions,
            )
            self.server.scene.add_label(
                f"/contact_points_{postfix}/label/{joint_name}",
                text=joint_name,
                position=positions,
            )

    def run(self):
        """Run the visualization server."""
        print(f"Dataset has {len(self.dataset)} samples")
        self._load_sample_data(0)
        self._update_visualization()
        self.run_server_loop()


# ============================================================================
# Public API
# ============================================================================
def create_visualization(
    data_path: str, prediction_path: str, port: int = 8080, device: str = "cuda"
):
    """Create interactive Viser visualization for grasp predictions.

    Args:
        data_path: Path to dataset (e.g., '/root/data/dexgys').
        prediction_path: Path to predictions JSON file.
        port: Port for visualization server.
        device: Device for hand model computation ('cuda' or 'cpu').
    """
    if "dexonomy" in data_path:
        dataset = DexonomyPredictionDataset(data_path=data_path, prediction_path=prediction_path)
    elif "dexgys" in data_path:
        dataset = DexGYSPredictionDataset(data_path=data_path, prediction_path=prediction_path)
    else:
        raise ValueError(f"Unsupported dataset: {data_path}")

    viz = GraspVisualization(dataset, port, device)
    viz.run()


def main(
    data_path: str = "/root/data/dexgys",
    prediction_path: str = "test_output/final/predictions.json",
    port: int = 8080,
    device: str = "cuda",
):
    """Main entry point for the visualization script.

    Args:
        data_path: Path to DexGYS dataset.
        prediction_path: Path to predictions JSON file.
        port: Port for visualization server.
        device: Device for computation ('cuda' or 'cpu').
    """
    print(f"Loading predictions from: {prediction_path}")
    print(f"Data path: {data_path}")

    create_visualization(
        data_path=data_path, prediction_path=prediction_path, port=port, device=device
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
