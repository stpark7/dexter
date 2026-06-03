import json
import numbers
import os

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from csdf import compute_sdf, index_vertices_by_faces

MUJOCO_GEOM = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}


def to_int(x):
    # tensor/list/ndarray -> int 로 변환
    if hasattr(x, "item"):
        return int(x.item())
    if isinstance(x, list | tuple | np.ndarray) and len(x) == 1:
        return int(x[0])
    if isinstance(x, numbers.Integral):
        return int(x)
    return int(x)  # 마지막 시도


def farthest_point_sample(xyz, num_points):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        num_points: number of samples
    Return:
        centroids: sampled pointcloud index, [B, num_points]
    """
    device = xyz.device
    b, n, c = xyz.shape
    centroids = torch.zeros(b, num_points, dtype=torch.long).to(device)
    distance = torch.ones(b, n).to(device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long).to(device)
    batch_indices = torch.arange(b, dtype=torch.long).to(device)

    for i in range(num_points):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: [batch_size, 3] tensor of axis-angle rotations

    Returns:
        rotation_matrix: [batch_size, 3, 3] tensor of rotation matrices
    """
    batch_size = axis_angle.shape[0]
    device = axis_angle.device

    # Handle the case where axis_angle might be zero
    angle = torch.norm(axis_angle, dim=1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)

    # Handle zero angle case
    angle = angle.squeeze(-1)
    zero_angle = angle < 1e-8

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    one_minus_cos = 1 - cos_angle

    # Extract axis components
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    # Build rotation matrix using Rodrigues' formula
    rotation_matrix = torch.zeros(batch_size, 3, 3, device=device)

    # Diagonal elements
    rotation_matrix[:, 0, 0] = cos_angle + x * x * one_minus_cos
    rotation_matrix[:, 1, 1] = cos_angle + y * y * one_minus_cos
    rotation_matrix[:, 2, 2] = cos_angle + z * z * one_minus_cos

    # Off-diagonal elements
    rotation_matrix[:, 0, 1] = x * y * one_minus_cos - z * sin_angle
    rotation_matrix[:, 0, 2] = x * z * one_minus_cos + y * sin_angle
    rotation_matrix[:, 1, 0] = y * x * one_minus_cos + z * sin_angle
    rotation_matrix[:, 1, 2] = y * z * one_minus_cos - x * sin_angle
    rotation_matrix[:, 2, 0] = z * x * one_minus_cos - y * sin_angle
    rotation_matrix[:, 2, 1] = z * y * one_minus_cos + x * sin_angle

    # Handle zero angle case - should be identity matrix
    identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix = torch.where(zero_angle.unsqueeze(-1).unsqueeze(-1), identity, rotation_matrix)

    return rotation_matrix


def sample_points_from_mesh(vertices, faces, num_samples):
    import open3d as o3d

    """
    Sample points from mesh surface using open3d

    Args:
        vertices: [N, 3] tensor of vertices
        faces: [F, 3] tensor of face indices
        num_samples: number of points to sample

    Returns:
        sampled_points: [num_samples, 3] tensor of sampled points
    """
    # Convert to numpy for open3d
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy().astype(np.int32)

    # Create open3d mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)

    # Sample points
    sampled_pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    sampled_points = np.asarray(sampled_pcd.points)

    # Convert back to tensor
    return torch.tensor(sampled_points, dtype=vertices.dtype, device=vertices.device)


def min_distance_from_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    m_num = m.shape[-2]
    n_num = n.shape[-2]

    # m_: [..., M, N, 3]
    # n_: [..., M, N, 3]
    m_ = m.unsqueeze(-2)  # [..., M, 1, 3]
    n_ = n.unsqueeze(-3)  # [..., 1, N, 3]

    m_ = torch.repeat_interleave(m_, n_num, dim=-2)  # [..., M, N, 3]
    n_ = torch.repeat_interleave(n_, m_num, dim=-3)  # [..., M, N, 3]

    # [..., M, N]
    pairwise_dis = torch.sqrt(((m_ - n_) ** 2).sum(dim=-1))

    ret_dis = torch.min(pairwise_dis, dim=-1)[0]  # [..., M]

    return ret_dis


def soft_distance(distance):
    """
    :param distance: [..., M]
    :return: [..., M]
    """
    sigmoid = torch.nn.Sigmoid()
    normalize_factor = 60  # decided by visualization
    return 1 - 2 * (sigmoid(normalize_factor * distance) - 0.5)


def contact_map_of_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    distances = min_distance_from_m_to_n(m, n)  # [..., M]
    distances = soft_distance(distances)
    return distances


class ShadowHandModel:
    def __init__(self, base_dir: str, device: str = "cpu", vis: bool = False):
        mjcf_path = os.path.join(base_dir, "shadow_hand.xml" if not vis else "shadow_hand_vis.xml")
        mesh_path = os.path.join(base_dir, "meshes")
        contact_points_path = os.path.join(base_dir, "contact_points.json")
        penetration_points_path = os.path.join(base_dir, "penetration_points.json")
        fingertip_points_path = os.path.join(base_dir, "contact_points_tips.json")
        n_surface_points = 1024
        self.device = torch.device(device)
        with open(mjcf_path) as f:
            mjcf_content = f.read()
        self.chain = pk.build_chain_from_mjcf(mjcf_content).to(dtype=torch.float32, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        penetration_points = None
        if penetration_points_path is not None:
            with open(penetration_points_path) as f:
                penetration_points = json.load(f)
        contact_points = None
        if contact_points_path is not None:
            with open(contact_points_path) as f:
                contact_points = json.load(f)
        fingertip_points = None
        if fingertip_points_path is not None:
            with open(fingertip_points_path) as f:
                fingertip_points = json.load(f)

        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            if len(body.link.visuals) > 0:
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
                    if visual.geom_type == "box":
                        link_mesh = trimesh.load_mesh(
                            os.path.join(mesh_path, "box.obj"), process=False
                        )
                        link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                    elif visual.geom_type == "capsule":
                        # form 3.9 to 4.05 of trimesh
                        link_mesh = trimesh.primitives.Capsule(
                            radius=visual.geom_param[0], height=visual.geom_param[1] * 2
                        )  # .apply_translation((0, 0, -visual.geom_param[1]))
                    # elif visual.geom_type == "mesh":
                    else:
                        link_mesh = trimesh.load_mesh(
                            os.path.join(mesh_path, visual.geom_param[0].split(":")[1] + ".obj"),
                            process=False,
                        )
                        if visual.geom_param[1] is not None:
                            scale = torch.tensor(
                                visual.geom_param[1], dtype=torch.float32, device=device
                            )

                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float32, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                contact_candidates = (
                    torch.tensor(
                        contact_points[link_name], dtype=torch.float32, device=device
                    ).reshape(-1, 3)
                    if contact_points is not None
                    else None
                )
                penetration_keypoints = (
                    torch.tensor(
                        penetration_points[link_name],
                        dtype=torch.float32,
                        device=device,
                    ).reshape(-1, 3)
                    if penetration_points is not None
                    else None
                )
                fingertip_keypoints = (
                    torch.tensor(
                        fingertip_points[link_name], dtype=torch.float32, device=device
                    ).reshape(-1, 3)
                    if fingertip_points is not None and link_name in fingertip_points
                    else None
                )
                link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                self.mesh[link_name] = {
                    "vertices": link_vertices.float(),
                    "faces": link_faces,
                    "contact_candidates": contact_candidates.float(),
                    "penetration_keypoints": penetration_keypoints.float(),
                    "fingertip_keypoints": fingertip_keypoints.float(),
                    "face_verts": link_face_verts.float(),
                }
                if link_name not in ["robot0:palm", "robot0:lfmetacarpal_child"]:
                    self.mesh[link_name]["geom_param"] = body.link.visuals[0].geom_param
                areas[link_name] = trimesh.Trimesh(
                    link_vertices.cpu().float().numpy(), link_faces.cpu().numpy()
                ).area.item()
            for children in body.children:
                build_mesh_recurse(children)

        build_mesh_recurse(self.chain._root)  # noqa: SLF001

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)

        set_joint_range_recurse(self.chain._root)  # noqa: SLF001
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        total_area = sum(areas.values())
        num_samples = {
            link_name: int(areas[link_name] / total_area * n_surface_points)
            for link_name in self.mesh
        }
        num_samples["robot0:palm"] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float32, device=device
                ).reshape(0, 3)
                continue

            dense_point_cloud = sample_points_from_mesh(
                self.mesh[link_name]["vertices"],
                self.mesh[link_name]["faces"],
                100 * num_samples[link_name],
            ).unsqueeze(0)  # Add batch dimension for FPS

            # Use farthest point sampling to get final surface points
            fps_indices = farthest_point_sample(dense_point_cloud, num_samples[link_name])
            surface_points = dense_point_cloud[0, fps_indices[0]]
            surface_points.to(dtype=torch.float32, device=device)
            self.mesh[link_name]["surface_points"] = surface_points

    def __call__(
        self,
        hand_pose,
        object_pc=None,
        with_meshes=False,
        with_surface_points=False,
        with_contact_candidates=False,
        with_penetration_keypoints=False,
        with_penetration=False,
    ):
        hand_pose = hand_pose.to(self.device)
        if object_pc is not None:
            object_pc = object_pc.to(self.device)

        batch_size = len(hand_pose)
        global_translation = hand_pose[:, 0:3]
        global_rotation = axis_angle_to_matrix(hand_pose[:, 3:6])
        current_status = self.chain.forward_kinematics(hand_pose[:, 6:])

        hand = {}
        hand["hand_pose"] = hand_pose

        if object_pc is not None:
            distances = []
            penetration = []
            x = (
                object_pc - global_translation.unsqueeze(1)
            ) @ global_rotation  # (batch_size, num_samples, 3)
            for link_name in self.mesh:
                if link_name in [
                    "robot0:ffknuckle_child",
                    "robot0:mfknuckle_child",
                    "robot0:rfknuckle_child",
                    "robot0:lfknuckle_child",
                    "robot0:thbase_child",
                    "robot0:thhub_child",
                ]:
                    continue
                matrix = current_status[link_name].get_matrix()
                x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
                x_local = x_local.reshape(-1, 3)  # (batch_size * num_samples, 3)
                if "geom_param" not in self.mesh[link_name]:
                    face_verts = self.mesh[link_name]["face_verts"]
                    dis_local, _, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                    dis_local = dis_local * (-dis_signs)
                    if with_penetration:
                        penetration_local = dis_local
                else:
                    height = self.mesh[link_name]["geom_param"][1] * 2
                    radius = self.mesh[link_name]["geom_param"][0]
                    projected_point = x_local.detach().clone()
                    projected_point[:, :2] = 0

                    projected_point[:, 2] = torch.clamp(projected_point[:, 2], 0, height)
                    direction = torch.nn.functional.normalize(x_local.detach() - projected_point)
                    direction = torch.where(
                        direction.norm(dim=1, keepdim=True) < 0.9,
                        torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device),
                        direction,
                    )
                    nearest_point = projected_point + radius * direction
                    dis_local = (
                        (x_local - nearest_point).square().sum(dim=1)
                    )  # (batch_size * num_samples)
                    mask = (x_local.detach() - projected_point).norm(dim=1) < radius
                    dis_local = torch.where(mask, dis_local, -dis_local)
                    if with_penetration:
                        if link_name not in [
                            "robot0:thmiddle_child",
                            "robot0:thdistal_child",
                            "robot0:thproximal_child",
                        ]:
                            nearest_point = projected_point.clone()
                            nearest_point[:, 1] = -radius
                            penetration_local = (
                                (x_local - nearest_point).square().sum(dim=1)
                            )  # (batch_size * num_samples)
                            penetration_local = torch.where(
                                mask, penetration_local, -penetration_local
                            )
                        else:
                            nearest_point = projected_point.clone()
                            nearest_point[:, 0] = -radius
                            penetration_local = (
                                (x_local - nearest_point).square().sum(dim=1)
                            )  # (batch_size * num_samples)
                            penetration_local = torch.where(
                                mask, penetration_local, -penetration_local
                            )
                            # penetration_local = dis_local
                distances.append(
                    dis_local.reshape(x.shape[0], x.shape[1])
                )  # (batch_size, num_samples)
                if with_penetration:
                    penetration.append(penetration_local.reshape(x.shape[0], x.shape[1]))
            distances = torch.max(torch.stack(distances), dim=0)[0]
            hand["distances"] = distances
            if with_penetration:
                penetration = torch.max(torch.stack(penetration), dim=0)[0]
                hand["penetration"] = penetration

        def get_points(key):
            points = [
                current_status[link_name]
                .transform_points(self.mesh[link_name][key])
                .expand(batch_size, -1, -1)
                for link_name in self.mesh
            ]
            points = torch.concat(points, dim=1) @ global_rotation.transpose(
                1, 2
            ) + global_translation.unsqueeze(1)
            return points

        def get_points_per_link(key):
            points_per_link = {}
            for link_name in self.mesh:
                points_per_link[link_name] = (
                    current_status[link_name]
                    .transform_points(self.mesh[link_name][key])
                    .expand(batch_size, -1, -1)
                )
                points_per_link[link_name] = points_per_link[link_name] @ global_rotation.transpose(
                    1, 2
                ) + global_translation.unsqueeze(1)
            return points_per_link

        if with_meshes:
            hand["vertices"] = get_points("vertices")
            n_vertices = 0
            faces = []
            for link_name in self.mesh:
                faces.append(self.mesh[link_name]["faces"] + n_vertices)
                n_vertices += self.mesh[link_name]["vertices"].shape[0]
            hand["faces"] = torch.concat(faces)

        if with_surface_points:
            hand["surface_points"] = get_points("surface_points")
            hand["surface_points_per_link"] = get_points_per_link("surface_points")

        if with_contact_candidates:
            # b,object_pc.shape[1]
            # if object_pc is not None:
            #     dis_pred = pytorch3d.ops.knn_points(object_pc, get_points("contact_candidates")).dists[:, :, 0]
            #     hand["contact_candidates_dis"] = dis_pred
            hand["contact_candidates_dis"] = get_points("contact_candidates")

        if with_penetration_keypoints:
            hand["penetration_keypoints"] = get_points("penetration_keypoints")

        return hand
