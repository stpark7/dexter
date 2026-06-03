"""
Q1 grasp-quality metric for dexter DexGYS predictions.

Single-file orchestrator + worker (mirrors success_rate.py). The orchestrator stays light
(no torch/CUDA): it loads the predictions, splits them into ``--parallel`` slices,
and re-spawns this same file with ``--worker`` once per slice. Each slice is handed
to its worker over stdin, the worker loads ShadowHandModel/KaolinModel once and
evaluates its samples on the chosen GPU, and streams one ``@@RESULT@@`` line per
sample back on stdout. The orchestrator owns a single global rich progress bar,
aggregates the per-sample pen/valid_q1 values as they arrive, and prints the
summary. No temp prediction copy and no on-disk metric shards.

Ported from Grasp-as-You-Say (scripts/q1/eval_utils.py). Adapted to use
dexter's ShadowHandModel and dexter's OakInk mesh layout
(<mesh_root>/{oakink_shape_v2, metaV2, OakInkObjectsV2, OakInkVirtualObjectsV2}).

Per-sample input format matches scripts/test.py output:
    {"obj_id": str, "guidance": str, "predictions": list[list[float]]  # 28-D}

Usage:
    python -m benchmark.dexgys.q1 \
        --pred-path checkpoints/exp/predictions.json \
        --gpu 0 --parallel 4 \
        --mesh_root /datasets/dexgys/meshes \
        --assets_dir ./assets/shadowhand
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import random
import sys
from math import ceil
from statistics import mean
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from torch import Tensor

RESULT_MARKER = "@@RESULT@@"  # worker -> orchestrator per-sample result line prefix


def _heavy_import():
    """Import csdf/torch/pytorch3d/trimesh/scipy/dexter and bind them to module globals.

    Run only inside a worker process, never in the orchestrator, so the orchestrator
    stays light and GPU-free (CUDA_VISIBLE_DEVICES is set before this runs).
    """
    global np, torch, trimesh, scipy, csdf, T
    global compute_sdf, index_vertices_by_faces, ShadowHandModel
    import csdf  # noqa: F401
    import numpy as np  # noqa: F401
    import pytorch3d.transforms as T  # noqa: N812,F401
    import scipy.spatial  # noqa: F401
    import torch  # noqa: F401
    import trimesh  # noqa: F401
    from csdf import compute_sdf, index_vertices_by_faces  # noqa: F401

    from dexter.utils.shadowhand import ShadowHandModel  # noqa: F401


DEFAULT_Q1_CFG: Dict = {
    "lambda_torque": 10,
    "m": 8,
    "mu": 1,
    "nms": True,
    "thres_contact": 0.01,
    "thres_pen": 0.005,
    "thres_tpen": 0.01,
}


# ============================================================================
# Compute core (runs inside a worker)
# ============================================================================
class KaolinModel:
    """Loads an OakInk object mesh and computes SDF / surface samples for Q1."""

    def __init__(self, mesh_root: str, batch_size_each: int = 1, device: str = "cuda"):
        self.device = device
        self.batch_size_each = batch_size_each
        self.mesh_root = mesh_root

        meta_dir = os.path.join(mesh_root, "metaV2")
        with open(os.path.join(meta_dir, "object_id.json")) as f:
            self.real_meta = json.load(f)
        with open(os.path.join(meta_dir, "virtual_object_id.json")) as f:
            self.virtual_meta = json.load(f)

    def get_obj_path(self, oid: str, use_downsample: bool = True, key: str = "align") -> str:
        obj_suffix_path = "align_ds" if use_downsample else "align"
        if oid in self.real_meta:
            obj_name = self.real_meta[oid]["name"]
            obj_path = os.path.join(self.mesh_root, "OakInkObjectsV2")
        else:
            obj_name = self.virtual_meta[oid]["name"]
            obj_path = os.path.join(self.mesh_root, "OakInkVirtualObjectsV2")
        obj_mesh_path = list(
            glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj"))
            + glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply"))
        )
        if len(obj_mesh_path) > 1:
            obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
        assert len(obj_mesh_path) == 1, (len(obj_mesh_path), oid, obj_name)
        return obj_mesh_path[0]

    def initialize(self, object_id: str):
        obj_path = self.get_obj_path(object_id)
        obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
        bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
        self.object_face_verts_list = [
            index_vertices_by_faces(
                torch.tensor(obj_trimesh.vertices).to(self.device, torch.float32),
                torch.tensor(obj_trimesh.faces).to(self.device, torch.long),
            )
        ]
        self.surface_points_tensor = (
            torch.tensor(obj_trimesh.sample(4096))
            .to(dtype=torch.float32, device=self.device)
            .unsqueeze(0)
        )

    def cal_distance(self, x, with_closest_points: bool = False):
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance, normals, closest_points = [], [], []
        for i in range(x.shape[0]):
            face_verts = self.object_face_verts_list[i]
            dis, normal, dis_signs, _, _ = csdf.compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance).reshape(-1, n_points)
        normals = torch.stack(normals).reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = torch.stack(closest_points).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals


def cal_q1(cfg: Dict, hand_model, object_model, object_code: str, hand_pose: Tensor, device):
    """
    Q1 grasp metric for a single (object, hand pose).

    Params:
        cfg: q1 config dict (see DEFAULT_Q1_CFG)
        hand_model: dexter ShadowHandModel instance
        object_model: KaolinModel instance
        object_code: OakInk object id
        hand_pose: (28,) tensor — translation(3) + axis-angle(3) + joints(22)
    """
    object_model.initialize(object_code)
    object_model.batch_size_each = 1

    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = T.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])

    contact_points_object = []
    contact_normals = []
    for link_name in hand_model.mesh:
        if len(hand_model.mesh[link_name]["surface_points"]) == 0:
            continue
        surface_points = current_status[link_name].transform_points(
            hand_model.mesh[link_name]["surface_points"]
        )
        surface_points = surface_points @ global_rotation.transpose(
            1, 2
        ) + global_translation.unsqueeze(1)
        distances, normals, closest_points = object_model.cal_distance(
            surface_points, with_closest_points=True
        )
        if cfg["nms"]:
            nearest_point_index = distances.argmax()
            if -distances[0, nearest_point_index] < cfg["thres_contact"]:
                contact_points_object.append(closest_points[0, nearest_point_index])
                contact_normals.append(normals[0, nearest_point_index])
        else:
            contact_idx = (-distances < cfg["thres_contact"]).nonzero().reshape(-1)
            for idx in contact_idx:
                contact_points_object.append(closest_points[0, idx])
                contact_normals.append(normals[0, idx])

    if len(contact_points_object) == 0:
        contact_points_object.append(torch.tensor([0, 0, 0], dtype=torch.float, device=device))
        contact_normals.append(torch.tensor([1, 0, 0], dtype=torch.float, device=device))

    contact_points_object = torch.stack(contact_points_object).cpu().numpy()
    contact_normals = torch.stack(contact_normals).cpu().numpy()
    n_contact = len(contact_points_object)

    if np.isnan(contact_points_object).any() or np.isnan(contact_normals).any():
        return 0

    u1 = np.stack(
        [-contact_normals[:, 1], contact_normals[:, 0], np.zeros([n_contact], dtype=np.float32)],
        axis=1,
    )
    u2 = np.stack(
        [
            np.ones([n_contact], dtype=np.float32),
            np.zeros([n_contact], dtype=np.float32),
            np.zeros([n_contact], dtype=np.float32),
        ],
        axis=1,
    )
    u = np.where(np.linalg.norm(u1, axis=1, keepdims=True) > 1e-8, u1, u2)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(u, contact_normals)
    theta = np.linspace(0, 2 * np.pi, cfg["m"], endpoint=False).reshape(cfg["m"], 1, 1)
    contact_forces = (
        contact_normals + cfg["mu"] * (np.cos(theta) * u + np.sin(theta) * v)
    ).reshape(-1, 3)

    origin = np.array([0, 0, 0], dtype=np.float32)
    wrenches = np.concatenate(
        [
            np.concatenate(
                [
                    contact_forces,
                    cfg["lambda_torque"]
                    * np.cross(
                        np.tile(contact_points_object - origin, (cfg["m"], 1)), contact_forces
                    ),
                ],
                axis=1,
            ),
            np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32),
        ],
        axis=0,
    )
    try:
        wrench_space = scipy.spatial.ConvexHull(wrenches)
    except scipy.spatial._qhull.QhullError:
        return 0
    q1 = np.array([1], dtype=np.float32)
    for equation in wrench_space.equations:
        q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))
    return q1.item()


def cal_pen(hand_model, object_model, object_code: str, hand_pose: Tensor, device):
    object_model.initialize(object_code)
    object_model.batch_size_each = 1

    object_surface_points = object_model.surface_points_tensor
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = T.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])

    skip_links = {
        "robot0:forearm",
        "robot0:wrist_child",
        "robot0:ffknuckle_child",
        "robot0:mfknuckle_child",
        "robot0:rfknuckle_child",
        "robot0:lfknuckle_child",
        "robot0:thbase_child",
        "robot0:thhub_child",
    }

    distances = []
    x = (object_surface_points - global_translation.unsqueeze(1)) @ global_rotation
    for link_name in hand_model.mesh:
        if link_name in skip_links:
            continue
        matrix = current_status[link_name].get_matrix()
        x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
        x_local = x_local.reshape(-1, 3)
        if "geom_param" not in hand_model.mesh[link_name]:
            face_verts = hand_model.mesh[link_name]["face_verts"]
            dis_local, _, dis_signs, _, _ = compute_sdf(x_local, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local * (-dis_signs)
        else:
            height = hand_model.mesh[link_name]["geom_param"][1] * 2
            radius = hand_model.mesh[link_name]["geom_param"][0]
            nearest_point = x_local.detach().clone()
            nearest_point[:, :2] = 0
            nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
            dis_local = radius - (x_local - nearest_point).norm(dim=1)
        distances.append(dis_local.reshape(x.shape[0], x.shape[1]))
    distances = torch.max(torch.stack(distances, dim=0), dim=0)[0]
    return max(distances.max().item(), 0)


# ============================================================================
# Worker (one slice; loads models once, streams per-sample results on stdout)
# ============================================================================
def run_worker(gpu: int, mesh_root: str, assets_dir: str):
    """Evaluate the slice of samples read from stdin on one GPU.

    Loads ShadowHandModel/KaolinModel exactly once and reuses them across every
    sample in the slice. Emits one ``@@RESULT@@`` line per sample (object) so the
    orchestrator can advance its progress bar and aggregate incrementally.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Many workers share one GPU and the real work is on the GPU (SDF kernels), so cap
    # each worker to a single CPU thread. Otherwise every worker's torch/OpenMP/BLAS
    # runtime spawns one thread per core, and parallel * cores threads created at once
    # exhausts the process limit ("libgomp: Thread creation failed"). Set before the
    # heavy import so the OpenMP runtime picks it up at init.
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    _heavy_import()
    torch.set_num_threads(1)

    results = json.loads(sys.stdin.read())
    cfg = DEFAULT_Q1_CFG
    hand_model = ShadowHandModel(base_dir=assets_dir, device="cuda")
    object_model = KaolinModel(mesh_root, batch_size_each=1, device="cuda")

    for res in results:
        object_code = res["obj_id"]
        hand_pose = torch.tensor(res["predictions"], device="cuda")
        if hand_pose.dim() == 3:
            hand_pose = hand_pose.squeeze(1)
        elif hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)

        pen_values: list = []
        valid_q1_values: list = []
        for i in range(hand_pose.size(0)):
            q1 = cal_q1(cfg, hand_model, object_model, object_code, hand_pose[i], "cuda")
            pen = cal_pen(hand_model, object_model, object_code, hand_pose[i], "cuda")
            valid_q1 = q1 if pen < cfg["thres_pen"] else 0
            pen_values.append(pen)
            valid_q1_values.append(valid_q1)

        rec = {"obj_id": object_code, "pen": pen_values, "valid_q1": valid_q1_values}
        sys.stdout.write(RESULT_MARKER + json.dumps(rec) + "\n")
        sys.stdout.flush()


# ============================================================================
# Orchestrator (light; splits work, owns the rich progress bar, aggregates)
# ============================================================================
def random_select_scales(results):
    original_len = len(results)
    grouped: Dict[str, list] = {}
    for res in results:
        grouped.setdefault(res["obj_id"], []).append(res)
    picked = [group[random.randint(0, len(group) - 1)] for group in grouped.values()]
    print(f"Selected {len(picked)} / {original_len} results to evaluate.")
    return picked


def orchestrate(
    pred_path: str,
    gpu: int,
    parallel: int,
    partial_scales: bool,
    mesh_root: str,
    assets_dir: str,
):
    """Split predictions into `parallel` slices, run one worker subprocess per slice
    (sharing the GPU), and render a single global rich progress bar while aggregating
    the per-sample pen/valid_q1 values streamed back by the workers."""
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        sys.exit("This benchmark needs 'rich' (pip install rich).")

    with open(pred_path) as rf:
        results = json.load(rf)
    if partial_scales:
        results = random_select_scales(results)
    total = len(results)
    if total == 0:
        sys.exit("No predictions to evaluate.")

    parallel = min(parallel, total)
    per_proc = ceil(total / parallel)
    slices = [results[i : i + per_proc] for i in range(0, total, per_proc)]
    print(f"Evaluating {total} samples across {len(slices)} workers on GPU {gpu}.")

    worker_cmd = [
        sys.executable,
        osp.abspath(__file__),
        "--worker",
        "--gpu",
        str(gpu),
        "--mesh_root",
        mesh_root,
        "--assets_dir",
        assets_dir,
    ]

    pen_values: list = []
    valid_q1_values: list = []
    failed: list = []
    lock = threading.Lock()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• pen {task.fields[pen]:.4f} • valid_q1 {task.fields[valid_q1]:.4f} •"),
        TimeElapsedColumn(),
        TextColumn("elapsed •"),
        TimeRemainingColumn(),
        TextColumn("left"),
    )

    def run_slice(slice_idx: int, samples: list):
        # Drain stderr in a side thread so a chatty worker can't deadlock on a full pipe.
        proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_buf: list = []
        drainer = threading.Thread(target=lambda: stderr_buf.append(proc.stderr.read()), daemon=True)
        drainer.start()

        proc.stdin.write(json.dumps(samples))
        proc.stdin.close()
        for line in proc.stdout:
            if not line.startswith(RESULT_MARKER):
                continue  # ignore stray library output; only act on result markers
            rec = json.loads(line[len(RESULT_MARKER) :])
            with lock:
                pen_values.extend(rec["pen"])
                valid_q1_values.extend(rec["valid_q1"])
                progress.update(
                    task,
                    advance=1,
                    pen=mean(pen_values),
                    valid_q1=mean(valid_q1_values),
                )

        proc.wait()
        drainer.join()
        if proc.returncode != 0:
            tail = ((stderr_buf[0] if stderr_buf else "") or "").strip().splitlines()[-1:] or [""]
            with lock:
                failed.append(slice_idx)
                progress.console.print(f"[red]worker {slice_idx} FAILED[/] {tail[0]}")

    with progress:
        task = progress.add_task("Evaluating", total=total, pen=0.0, valid_q1=0.0)
        with ThreadPoolExecutor(max_workers=len(slices)) as ex:
            for fut in as_completed([ex.submit(run_slice, i, s) for i, s in enumerate(slices)]):
                fut.result()

    if pen_values:
        print(f"mean overall pen: {mean(pen_values):.6f}")
        print(f"valid_q1:         {mean(valid_q1_values):.6f}")
    if failed:
        print(f"{len(failed)} worker(s) failed: {sorted(failed)}")


def main(
    pred_path: str | None = None,
    gpu: int = 0,
    parallel: int = 1,
    partial_scales: bool = False,
    num_scales: int | None = None,
    mesh_root: str = "/datasets/dexgys/meshes",
    assets_dir: str = "./assets/shadowhand",
    worker: bool = False,
):
    """Compute the Q1 grasp-quality metric for a predictions.json.

    Orchestrator mode (default): split Q1 evaluation across --parallel worker
    subprocesses sharing one GPU and aggregate the streamed results under a single
    rich progress bar.
    Worker mode (--worker, internal): evaluate the slice of samples read from stdin
    on the GPU and stream one @@RESULT@@ line per sample. Not for direct use.

    Args:
        pred_path: path to predictions.json from scripts/test.py.
        gpu: CUDA device id all workers run on (sets CUDA_VISIBLE_DEVICES).
        parallel: number of concurrent worker subprocesses sharing the GPU.
        partial_scales: randomly subsample one prediction per object before eval.
        num_scales: (unused; kept for parity with the original CLI).
        mesh_root: dir with metaV2/, OakInkObjectsV2/, OakInkVirtualObjectsV2/.
        assets_dir: dexter ShadowHandModel base_dir.
        worker: internal — evaluate one stdin slice on the GPU (set by the orchestrator).
    """
    if worker:
        run_worker(gpu, mesh_root, assets_dir)
        return

    if pred_path is None:
        sys.exit("--pred-path is required (predictions.json)")
    orchestrate(pred_path, gpu, parallel, partial_scales, mesh_root, assets_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
