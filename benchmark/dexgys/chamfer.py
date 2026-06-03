import json
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import viser
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from scipy.optimize import linear_sum_assignment
from torch.functional import Tensor
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat

from dexter.models.loss import (
    get_cmap_loss,
    get_hand_chamfer_loss,
    get_obj_penetration_loss,
    get_self_penetration_loss,
)
from dexter.utils.shadowhand import ShadowHandModel


class MetricsDashboard:
    """Real-time TUI dashboard for displaying evaluation metrics."""

    def __init__(self):
        self.console = Console()
        self.results = []
        self.metric_names = [
            "hand_chamfer_loss",
            "cmap_loss",
            "obj_penetration_loss",
            "self_penetration_loss",
        ]

    def create_layout(self, current_result: dict, progress: Progress, task_id) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=3),
            Layout(name="current", size=8),
            Layout(name="stats", size=12),
        )

        # Header
        layout["header"].update(
            Panel("[bold cyan]DexGYS Evaluation Dashboard[/bold cyan]", style="cyan")
        )

        # Progress bar
        layout["progress"].update(Panel(progress))

        # Current sample info
        if current_result:
            current_table = Table(show_header=False, box=None, padding=(0, 1))
            current_table.add_column("Key", style="bold yellow")
            current_table.add_column("Value")
            current_table.add_row("Scene", current_result["scene_name"])
            current_table.add_row("Prompt", current_result["prompt"])
            current_table.add_row(
                "Match", f"{current_result['match_idx']}/{current_result['num_gt']}"
            )
            layout["current"].update(Panel(current_table, title="[bold]Current Sample[/bold]"))
        else:
            layout["current"].update(
                Panel("Waiting for first sample...", title="[bold]Current Sample[/bold]")
            )

        # Running statistics
        if len(self.results) > 0:
            stats_table = Table(show_header=True, box=None, padding=(0, 1))
            stats_table.add_column("Metric", style="bold green")
            stats_table.add_column("Current", justify="right", style="yellow")
            stats_table.add_column("Mean", justify="right", style="cyan")
            stats_table.add_column("Std", justify="right", style="magenta")
            stats_table.add_column("Min", justify="right", style="blue")
            stats_table.add_column("Max", justify="right", style="red")

            df = pd.DataFrame(self.results)
            for metric in self.metric_names:
                if metric in df.columns:
                    current_val = current_result.get(metric, 0.0)
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    min_val = df[metric].min()
                    max_val = df[metric].max()
                    stats_table.add_row(
                        metric,
                        f"{current_val:.4f}",
                        f"{mean_val:.4f}",
                        f"{std_val:.4f}",
                        f"{min_val:.4f}",
                        f"{max_val:.4f}",
                    )

            layout["stats"].update(Panel(stats_table, title="[bold]Statistics[/bold]"))
        else:
            layout["stats"].update(Panel("No statistics yet...", title="[bold]Statistics[/bold]"))

        return layout

    def add_result(self, result: dict):
        """Add a new result to the tracked metrics."""
        self.results.append(result)


class Matcher(nn.Module):
    def __init__(self, weight_dict: Dict[str, float], rotation_type: str = "axis_angle"):
        super().__init__()
        self.weight_dict = {k: v for k, v in weight_dict.items() if v > 0}
        self.rotation_type = rotation_type

    @torch.no_grad()
    def forward(self, preds: Tensor, targets: Tensor):
        cost_matrices = []
        for name, weight in self.weight_dict.items():
            m = getattr(self, f"get_{name}_cost_mat")
            cost_mat = m(preds, targets, weight=weight)
            cost_matrices.append(cost_mat)
        final_cost = torch.stack(cost_matrices).sum(dim=0)

        assign = linear_sum_assignment(final_cost.cpu().numpy())

        return {
            "final_cost": final_cost,
            "match_idx": assign[1].item(),
            "cost_matrices": cost_matrices,
        }

    def get_hand_mesh_cost_mat(
        self,
        prediction: Tensor,
        targets: Tensor,
        weight: float = 1.0,
    ) -> List[Tensor]:
        # TODO: implement chamfer loss for hand mesh cost
        raise NotImplementedError(
            "Unable to calculate hand mesh cost matrix yet. Please help me to implement it ^_^"
        )

    def get_qpos_cost_mat(
        self,
        prediction: Tensor,
        targets: Tensor,
        weight: float = 1.0,
    ) -> Tensor:
        pred_qpos = prediction[..., 6:]
        target_qpos = targets[..., 6:]
        return self._get_cost_mat_by_elementwise(pred_qpos, target_qpos, weight=weight)

    def get_translation_cost_mat(
        self,
        prediction: Tensor,
        targets: Tensor,
        weight: float = 1.0,
    ) -> Tensor:
        pred_t = prediction[..., :3]
        target_t = targets[..., :3]
        return self._get_cost_mat_by_elementwise(pred_t, target_t, weight=weight)

    def get_rotation_cost_mat(
        self,
        prediction: Tensor,
        targets: List[Tensor],
        weight: float = 1.0,
    ) -> List[Tensor]:
        rotation_type = self.rotation_type
        if hasattr(self, f"_get_{rotation_type}_cost_mat"):
            m = getattr(self, f"_get_{rotation_type}_cost_mat")
            pred_r = prediction[..., 3:6]
            target_r = targets[..., 3:6]
            return m(pred_r, target_r, weight)
        else:
            raise NotImplementedError(f"Unable to get {rotation_type} cost matrix")

    def _get_cost_mat_by_elementwise(
        self,
        prediction: Tensor,
        targets: Tensor,
        weight: float = 1.0,
        element_wise_func: Callable[[Tensor, Tensor], Tensor] = partial(
            F.l1_loss, reduction="none"
        ),
    ) -> Tensor:
        if prediction.shape[0] == 1:
            prediction = prediction.repeat(targets.shape[0], 1)
        cost = element_wise_func(prediction, targets)
        cost = cost.sum(dim=-1).reshape(-1, cost.shape[0])
        return weight * cost

    def _get_quaternion_cost_mat(
        self, prediction: Tensor, targets: Tensor, weight: float = 1.0
    ) -> Tensor:
        cost = 1 - (prediction @ targets.T).abs().detach()
        return weight * cost

    def _get_rotation_6d_cost_mat(
        self, prediction: Tensor, targets: Tensor, weight: float = 1.0
    ) -> List[Tensor]:
        cost_mat = self._get_cost_mat_by_elementwise(
            prediction,
            targets,
            weight=weight,
        )
        return cost_mat

    def _get_euler_cost_mat(
        self, prediction: Tensor, targets: Tensor, weight: float = 1.0
    ) -> Tensor:
        """
        specially-designed l1 loss for euler angles
        """
        error = (prediction - targets).abs().sum(-1)
        cost = torch.where(error < 0.5, error, 1 - error)
        return weight * cost

    def _get_axis_angle_cost_mat(
        self, prediction: Tensor, targets: List[Tensor], weight: float = 1.0
    ) -> List[Tensor]:
        def axangle2quat(axangle: Tensor) -> Tensor:
            return torch.from_numpy(
                mat2quat(axangle2mat(axangle, torch.norm(axangle, dim=-1, keepdim=True)))
            ).float()

        pred_q = torch.stack([axangle2quat(p) for p in prediction])
        gt_q = torch.stack([axangle2quat(t) for t in targets])
        return self._get_quaternion_cost_mat(pred_q, gt_q, weight)


def load_metadata(data_path: str):
    with open(data_path) as f:
        all_data = json.load(f)

    obj_dict = {}
    for i in range(len(all_data)):
        obj_code = ".".join(
            [
                str(all_data[i]["cate_id"]),
                str(all_data[i]["obj_id"]),
                str(all_data[i]["action_id"]),
            ]
        )
        if obj_code in obj_dict:
            obj_dict[obj_code]["dex_grasp"].append(all_data[i]["dex_grasp"])
            obj_dict[obj_code]["guidance"].append(all_data[i]["guidance"])
        else:
            obj_dict[obj_code] = all_data[i]
            obj_dict[obj_code]["dex_grasp"] = [obj_dict[obj_code]["dex_grasp"]]
            obj_dict[obj_code]["guidance"] = [obj_dict[obj_code]["guidance"]]
    return obj_dict


def count_valid_predictions(pred_path: str, data_path: str = "/root/data/dexgys"):
    """Count the number of valid predictions that will be processed."""
    data_path = Path(data_path)
    metadata = load_metadata(data_path / "test.json")
    keys = metadata.keys()

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    valid_count = 0
    for pred in predictions:
        scene_name = pred["obj_id"]
        prompt = pred["guidance"]

        matched_keys = list(filter(lambda x: scene_name in x, keys))
        if len(matched_keys) == 0:
            continue

        target_keys = []
        for k in matched_keys:
            prompts = metadata[k]["guidance"]
            if prompt in prompts:
                target_keys.append(k)

        if len(target_keys) != 1:
            continue

        valid_count += 1

    return valid_count


def load_grasps_iterator(
    pred_path: str,
    data_path: str = "/root/data/dexgys",
    rotation_type: str = "axis_angle",
    weight_qpos: float = 1.0,
    weight_translation: float = 2.0,
    weight_rotation: float = 2.0,
    visualize: bool = False,
):
    """
    Iterator that yields grasp information for visualization and benchmarking.

    Yields:
        dict containing scene info, predictions, targets, and matched indices
    """
    data_path = Path(data_path)
    metadata = load_metadata(data_path / "test.json")
    keys = metadata.keys()

    # Initialize matcher
    weight_dict = {
        "qpos": weight_qpos,
        "translation": weight_translation,
        "rotation": weight_rotation,
    }
    matcher = Matcher(weight_dict)

    # Initialize Hand Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hand_model = ShadowHandModel(base_dir="./assets/shadowhand", device=device)

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    for pred in predictions:
        # for scene_id in scene_ids:
        obj_path = data_path / "data" / pred["obj_id"] / "xyzc.npy"
        obj_pc = np.load(obj_path)[..., :3]
        prompt = pred["guidance"]
        pred_grasp = np.array(pred["predictions"])
        scene_name = pred["obj_id"]

        # with h5py.File(pred_path, "r") as f:
        #     obj_pc = f[scene_id]["pointcloud"][:].astype(np.float32)
        #     pred = f[scene_id]["pred"][:].astype(np.float32)
        #     prompt = f[scene_id]["prompt"][()].decode("utf-8")
        #     scene_name = f[scene_id]["scene_name"][()].decode("utf-8")

        return_dict = dict(scene_name=scene_name)

        matched_keys = list(filter(lambda x: scene_name in x, keys))
        if len(matched_keys) == 0:
            print(f"Scene {scene_name} not found in metadata")
            continue

        target_keys = []
        for k in matched_keys:
            prompts = metadata[k]["guidance"]
            if prompt in prompts:
                target_keys.append(k)

        if len(target_keys) != 1:
            print(f"Scene {scene_name} has {len(target_keys)} target keys")
            continue

        target_key = target_keys[0]
        gt_grasps = metadata[target_key]["dex_grasp"]

        # Convert to torch tensors
        pred_grasp = torch.from_numpy(pred_grasp).float().unsqueeze(0)  # (1, D)
        gt = torch.from_numpy(np.array(gt_grasps)).float()  # (ngt, D)

        # match
        match_result = matcher(pred_grasp, gt)
        target_idx = match_result["match_idx"]

        pred_grasp = pred_grasp.to(device)
        gt = gt.to(device)
        obj_pc_tensor = torch.from_numpy(obj_pc[:, :3]).float().to(device)

        # Compute all target hands
        all_gt_hands = []
        if visualize:
            all_gt_hands = hand_model(
                gt,
                obj_pc_tensor,
                with_meshes=True,
                with_penetration=True,
                with_surface_points=True,
                with_penetration_keypoints=True,
            )

        # Compute predicted hand
        pred_hand = hand_model(
            pred_grasp,
            obj_pc_tensor,
            with_meshes=True,
            with_penetration=True,
            with_surface_points=True,
            with_penetration_keypoints=True,
        )

        # Compute matched target hand for metrics
        gt_hand = hand_model(
            gt[target_idx : target_idx + 1],
            obj_pc_tensor,
            with_meshes=True,
            with_penetration=True,
            with_surface_points=True,
            with_penetration_keypoints=True,
        )
        gt_hand["obj_pc"] = obj_pc_tensor

        # Compute losses
        hand_chamfer_loss = get_hand_chamfer_loss(pred_hand, gt_hand, reduce=False)
        cmap_loss = get_cmap_loss(pred_hand, gt_hand, reduce=False)
        obj_penetration_loss = get_obj_penetration_loss(
            pred_hand, gt_hand, training=False, reduce=False
        )
        self_penetration_loss = get_self_penetration_loss(
            pred_hand, gt_hand, training=False, reduce=False
        )

        return_dict["prompt"] = prompt
        return_dict["obj_pc"] = obj_pc_tensor
        return_dict["pred_hand"] = pred_hand
        return_dict["gt_hand"] = gt_hand
        return_dict["all_gt_hands"] = all_gt_hands
        return_dict["target_idx"] = target_idx
        return_dict["num_gt"] = len(gt_grasps)
        return_dict["metrics"] = {
            "hand_chamfer_loss": hand_chamfer_loss.item(),
            "cmap_loss": cmap_loss.item(),
            "obj_penetration_loss": obj_penetration_loss.item(),
            "self_penetration_loss": self_penetration_loss.item(),
        }
        return_dict["cost"] = match_result["final_cost"][0].squeeze()
        return_dict["cost_matrices"] = match_result["cost_matrices"]

        yield return_dict


def visualize_grasps(grasp_info: dict, server: viser.ViserServer):
    """
    Visualize grasp information using viser.

    Args:
        grasp_info: Dictionary containing scene info and hand meshes
        server: Viser server instance
    """
    # Clear previous scene
    server.scene.reset()

    # Visualize object point cloud
    server.scene.add_point_cloud(
        "/object_pc",
        points=grasp_info["obj_pc"].cpu().numpy(),
        colors=np.array([150, 150, 150]),
        point_size=0.001,
    )

    # Visualize all target hands
    for i in range(len(grasp_info["all_gt_hands"]["vertices"])):
        # Different color for selected target vs other targets
        if i == grasp_info["target_idx"]:
            color = (0, 255, 0)  # Green for selected target
        else:
            color = (100, 100, 255)  # Blue for other targets

        server.scene.add_mesh_simple(
            f"/target_hand_{i}_cost{grasp_info['cost'][i]:.3f}",
            vertices=grasp_info["all_gt_hands"]["vertices"][i].cpu().numpy(),
            faces=grasp_info["all_gt_hands"]["faces"].cpu().numpy(),
            color=color,
            visible=False,
        )

    # Visualize predicted hand
    server.scene.add_mesh_simple(
        "/pred_hand",
        vertices=grasp_info["pred_hand"]["vertices"].cpu().numpy(),
        faces=grasp_info["pred_hand"]["faces"].cpu().numpy(),
        color=(255, 0, 0),  # Red for prediction
    )

    print(f"\nVisualization: {grasp_info['scene_id']}")
    print(f"Scene: {grasp_info['scene_name']}")
    print(f"Prompt: {grasp_info['prompt']}")
    print(f"Target idx: {grasp_info['target_idx']}/{grasp_info['num_gt']}")
    print(f"Metrics: {grasp_info['metrics']}")


def benchmark(
    pred_path: str,
    data_path: str = "/root/data/dexgys",
    rotation_type: str = "axis_angle",
    weight_qpos: float = 1.0,
    weight_translation: float = 2.0,
    weight_rotation: float = 2.0,
    visualize: bool = False,
    use_dashboard: bool = True,
):
    """
    Benchmark predictions against ground truth using the Matcher.

    Args:
        pred_path: Path to HDF5 file with predictions
        data_path: Path to JSON file with metadata
        rotation_type: Type of rotation representation (e.g., 'axis_angle', 'quaternion', 'rotation_6d', 'euler')
        weight_qpos: Weight for qpos cost
        weight_translation: Weight for translation cost
        weight_rotation: Weight for rotation cost
        visualize: Whether to visualize grasps with viser
        use_dashboard: Whether to use the TUI dashboard (default: True)
    """
    # Initialize viser server if visualizing
    server = None
    if visualize:
        server = viser.ViserServer(port=8080)
        print("Visualization server running at http://localhost:8080")

    # Count total predictions for progress bar
    total_predictions = count_valid_predictions(pred_path, data_path)

    # Create iterator for loading grasps
    grasp_iterator = load_grasps_iterator(
        pred_path=pred_path,
        data_path=data_path,
        rotation_type=rotation_type,
        weight_qpos=weight_qpos,
        weight_translation=weight_translation,
        weight_rotation=weight_rotation,
        visualize=visualize,
    )

    results = []

    if use_dashboard and not visualize:
        # Use rich dashboard
        dashboard = MetricsDashboard()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("[cyan]Evaluating...", total=total_predictions)

        with Live(dashboard.create_layout(None, progress, task_id), refresh_per_second=4) as live:
            for grasp_info in grasp_iterator:
                # Collect results
                result = {
                    "scene_name": grasp_info["scene_name"],
                    "num_gt": grasp_info["num_gt"],
                    "match_idx": grasp_info["target_idx"],
                    "prompt": grasp_info["prompt"],
                    **grasp_info["metrics"],
                }
                results.append(result)
                dashboard.add_result(result)

                # Update dashboard
                progress.update(task_id, advance=1)
                live.update(dashboard.create_layout(result, progress, task_id))

    else:
        # Use simple tqdm progress bar with printing
        for grasp_info in tqdm.tqdm(grasp_iterator, total=total_predictions, desc="Evaluating"):
            # Visualize if requested
            if visualize and server is not None:
                visualize_grasps(grasp_info, server)
                print("\nPress Enter to continue to next scene...")
                input()

            # Collect results
            result = {
                "scene_name": grasp_info["scene_name"],
                "num_gt": grasp_info["num_gt"],
                "match_idx": grasp_info["target_idx"],
                "prompt": grasp_info["prompt"],
                **grasp_info["metrics"],
            }
            results.append(result)

            # Print progress
            columns = [
                "scene_name",
                "num_gt",
                "match_idx",
                "prompt",
                "hand_chamfer_loss",
                "cmap_loss",
                "obj_penetration_loss",
                "self_penetration_loss",
            ]
            msg = " | ".join(
                [
                    f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in result.items()
                    if k in columns
                ]
            )
            print(f"{msg}")

    # Stop server if running
    if server is not None:
        server.stop()

    # Save and print results
    results_df = pd.DataFrame(results)
    results_summary = results_df.select_dtypes(include=[np.number]).mean(axis=0)
    results_df.to_csv(Path(pred_path).parent / "benchmark.csv")
    print("\n" + "=" * 80)
    print("Benchmark Results Summary:")
    print("=" * 80)
    print(results_summary)
    print("=" * 80)


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
