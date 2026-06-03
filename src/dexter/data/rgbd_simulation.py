"""RGB-D simulation utilities for partial point cloud observation.

This module provides functions to simulate RGB-D camera observations from
full point clouds, including:
- Virtual camera placement based on grasp pose
- Hidden Point Removal (HPR) for visibility determination
- Sensor noise simulation (depth noise, outliers)
- Transform class for integration with data pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

# Type alias for data dict
DataDict = dict[str, Any]


def hidden_point_removal(
    points: np.ndarray,
    camera_position: np.ndarray,
    radius_factor: float = 100.0,
) -> np.ndarray:
    """Determine visible points using Hidden Point Removal (HPR) algorithm.

    Implementation of Katz, Tal, and Basri 2007:
    "Direct Visibility of Point Sets"

    Args:
        points: Point cloud, shape (N, 3) or (N, C) where C >= 3
        camera_position: Camera position, shape (3,)
        radius_factor: Factor to compute flipping radius. Higher values mean
            more conservative visibility (more points kept).

    Returns:
        Boolean mask of visible points, shape (N,)
    """
    # Extract XYZ coordinates
    xyz = points[:, :3]

    # Translate so camera is at origin
    translated = xyz - camera_position

    # Compute distances from camera
    distances = np.linalg.norm(translated, axis=1)

    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)

    # Compute flipping radius (should be larger than max distance)
    R = radius_factor * np.max(distances)

    # Spherical flip transformation
    # p' = p + 2(R - ||p||) * p / ||p||
    flip_factor = 2 * (R - distances) / distances
    flipped = translated + flip_factor[:, np.newaxis] * translated

    # Add camera position (origin after translation) to the point set
    flipped_with_camera = np.vstack([flipped, np.zeros(3)])

    # Compute convex hull
    try:
        hull = ConvexHull(flipped_with_camera)

        # Points on the hull are visible
        # hull.vertices gives indices of vertices on the convex hull
        visible_indices = set(hull.vertices)

        # Remove the camera point index (last point)
        visible_indices.discard(len(flipped))

        # Create visibility mask
        visibility_mask = np.zeros(len(points), dtype=bool)
        visibility_mask[list(visible_indices)] = True

    except Exception:
        # If convex hull fails (e.g., degenerate case), return all visible
        visibility_mask = np.ones(len(points), dtype=bool)

    return visibility_mask


def add_sensor_noise(
    points: np.ndarray,
    camera_position: np.ndarray | None = None,
    depth_noise_std: float = 0.002,
    lateral_noise_std: float = 0.001,
    outlier_ratio: float = 0.01,
    outlier_std: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Add realistic RGB-D sensor noise to point cloud.

    Simulates common depth sensor artifacts:
    1. Depth-dependent Gaussian noise (increases with distance)
    2. Lateral noise (perpendicular to viewing direction)
    3. Random outliers (flying pixels, multi-path interference)

    Args:
        points: Point cloud, shape (N, C) where C >= 3 (XYZ + optional RGB)
        camera_position: Camera position for depth-dependent noise.
            If None, uses origin (0, 0, 0).
        depth_noise_std: Standard deviation of depth noise in meters.
            Typical consumer RGB-D sensors: 1-3mm at 1m distance.
        lateral_noise_std: Standard deviation of lateral (XY) noise in meters.
        outlier_ratio: Fraction of points to corrupt as outliers.
        outlier_std: Standard deviation for outlier displacement.
        seed: Random seed for reproducibility.

    Returns:
        Noisy point cloud, shape (N, C)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    points = points.copy()
    n_points = len(points)
    xyz = points[:, :3]

    # Camera position defaults to origin
    if camera_position is None:
        camera_position = np.zeros(3)

    # Compute viewing directions and distances
    view_vectors = xyz - camera_position
    distances = np.linalg.norm(view_vectors, axis=1, keepdims=True)
    distances = np.maximum(distances, 1e-6)  # Avoid division by zero
    view_dirs = view_vectors / distances

    # 1. Depth-dependent noise (along viewing direction)
    # Noise increases linearly with distance (quadratic for real sensors, but linear is common approximation)
    depth_noise_scale = depth_noise_std * (distances.flatten() / 0.5)  # Normalized to 0.5m baseline
    depth_noise = rng.normal(0, 1, n_points) * depth_noise_scale
    xyz += depth_noise[:, np.newaxis] * view_dirs

    # 2. Lateral noise (perpendicular to viewing direction)
    # Generate random perpendicular directions
    random_vecs = rng.normal(0, 1, (n_points, 3))
    # Project out the viewing direction component
    lateral_dirs = random_vecs - np.sum(random_vecs * view_dirs, axis=1, keepdims=True) * view_dirs
    lateral_norms = np.linalg.norm(lateral_dirs, axis=1, keepdims=True)
    lateral_norms = np.maximum(lateral_norms, 1e-6)
    lateral_dirs = lateral_dirs / lateral_norms

    lateral_noise = rng.normal(0, lateral_noise_std, n_points)
    xyz += lateral_noise[:, np.newaxis] * lateral_dirs

    # 3. Outliers (flying pixels)
    n_outliers = int(n_points * outlier_ratio)
    if n_outliers > 0:
        outlier_indices = rng.choice(n_points, n_outliers, replace=False)
        outlier_displacement = rng.normal(0, outlier_std, (n_outliers, 3))
        xyz[outlier_indices] += outlier_displacement

    points[:, :3] = xyz
    return points


def compute_grasp_aware_cameras(
    grasp_pose: np.ndarray,
    pointcloud: np.ndarray,
    num_views: int = 3,
    camera_radius: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Compute camera positions that view the contact region.

    Cameras are placed on a hemisphere opposite to the grasp direction,
    ensuring the contact region between hand and object is visible.

    Args:
        grasp_pose: Shadow Hand pose (28,) where [0:3] is wrist translation
        pointcloud: Object point cloud (N, 6) with XYZ + RGB
        num_views: Number of camera viewpoints to generate
        camera_radius: Distance of cameras from object center
        seed: Random seed for reproducibility

    Returns:
        Array of camera positions, shape (num_views, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract wrist position and object centroid
    wrist_pos = grasp_pose[:3]
    obj_center = pointcloud[:, :3].mean(axis=0)

    # Compute grasp direction (from wrist toward object)
    grasp_dir = obj_center - wrist_pos
    grasp_dir_norm = np.linalg.norm(grasp_dir)
    if grasp_dir_norm < 1e-6:
        # Fallback if wrist is at object center
        grasp_dir = np.array([0.0, 0.0, 1.0])
    else:
        grasp_dir = grasp_dir / grasp_dir_norm

    # Camera direction is opposite to grasp direction
    # (cameras look at contact region from the other side)
    cam_dir = -grasp_dir

    # Build orthonormal basis for the hemisphere
    # Find a vector not parallel to cam_dir
    if abs(cam_dir[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    # Gram-Schmidt to get orthonormal basis
    right = np.cross(cam_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    up = up / np.linalg.norm(up)

    # Sample camera positions on hemisphere
    cameras = []
    for i in range(num_views):
        if num_views == 1:
            # Single view: place camera directly opposite to grasp
            theta = 0.0
            phi = 0.0
        else:
            # Multiple views: spread around hemisphere
            # theta: azimuth angle around cam_dir axis
            # phi: elevation from cam_dir (0 = aligned with cam_dir)
            theta = 2 * np.pi * i / num_views
            phi = np.random.uniform(0, np.pi / 4)  # Up to 45 degrees from center

        # Convert spherical to Cartesian in local frame
        # Local z = cam_dir, local x = right, local y = up
        local_x = np.sin(phi) * np.cos(theta)
        local_y = np.sin(phi) * np.sin(theta)
        local_z = np.cos(phi)

        # Transform to world coordinates
        cam_offset = local_x * right + local_y * up + local_z * cam_dir
        cam_pos = obj_center + camera_radius * cam_offset

        cameras.append(cam_pos)

    return np.array(cameras)


def compute_fixed_multiview_cameras(
    pointcloud: np.ndarray,
    camera_distance: float = 0.5,
    elevation: float = 0.3,
    views: list[str] | None = None,
) -> np.ndarray:
    """Compute fixed camera positions at canonical viewpoints.

    Standard setup used in RLBench, CALVIN, ManiSkill, and other benchmarks.
    Cameras are placed at fixed positions relative to the object/scene center.

    Available views:
        - "front": Looking from +Y axis toward object
        - "front_left": 45 degrees left of front
        - "front_right": 45 degrees right of front
        - "left": Looking from -X axis
        - "right": Looking from +X axis
        - "overhead": Looking from +Z axis (top-down)
        - "overhead_front": Elevated front view (45 deg from vertical)

    Args:
        pointcloud: Object point cloud (N, 6) with XYZ + RGB
        camera_distance: Distance from object center to camera
        elevation: Height offset for non-overhead cameras
        views: List of view names. If None, uses ["front", "left", "overhead"]

    Returns:
        Array of camera positions, shape (num_views, 3)
    """
    if views is None:
        views = ["front", "left", "overhead"]

    obj_center = pointcloud[:, :3].mean(axis=0)

    # Define canonical camera directions (unit vectors pointing FROM camera TO object)
    # Camera positions are obj_center - direction * distance
    view_directions = {
        "front": np.array([0.0, 1.0, 0.0]),  # Camera at +Y, looking at -Y
        "back": np.array([0.0, -1.0, 0.0]),  # Camera at -Y, looking at +Y
        "left": np.array([-1.0, 0.0, 0.0]),  # Camera at -X, looking at +X
        "right": np.array([1.0, 0.0, 0.0]),  # Camera at +X, looking at -X
        "front_left": np.array([-0.707, 0.707, 0.0]),
        "front_right": np.array([0.707, 0.707, 0.0]),
        "overhead": np.array([0.0, 0.0, 1.0]),  # Camera above, looking down
        "overhead_front": np.array([0.0, 0.5, 0.866]),  # 30 deg from vertical, front
        "overhead_left": np.array([-0.5, 0.0, 0.866]),
        "overhead_right": np.array([0.5, 0.0, 0.866]),
    }

    cameras = []
    for view in views:
        if view not in view_directions:
            raise ValueError(f"Unknown view: {view}. Available: {list(view_directions.keys())}")

        direction = view_directions[view]

        if view == "overhead":
            # Pure top-down, no lateral offset
            cam_pos = obj_center + np.array([0.0, 0.0, camera_distance])
        elif view.startswith("overhead_"):
            # Elevated views
            cam_pos = obj_center + direction * camera_distance
        else:
            # Horizontal views with elevation
            horizontal_dir = direction.copy()
            horizontal_dir[2] = 0
            horizontal_dir = horizontal_dir / (np.linalg.norm(horizontal_dir) + 1e-8)
            cam_pos = (
                obj_center - horizontal_dir * camera_distance + np.array([0.0, 0.0, elevation])
            )

        cameras.append(cam_pos)

    return np.array(cameras)


def compute_random_viewpoint_cameras(
    pointcloud: np.ndarray,
    num_views: int = 3,
    camera_distance_range: tuple[float, float] = (0.3, 0.6),
    elevation_range: tuple[float, float] = (0.1, 0.5),
    azimuth_range: tuple[float, float] = (0.0, 2 * np.pi),
    seed: int | None = None,
) -> np.ndarray:
    """Compute random camera positions for data augmentation.

    Samples camera positions uniformly in spherical coordinates around
    the object center. Useful for domain randomization and training
    viewpoint-invariant models.

    Args:
        pointcloud: Object point cloud (N, 6) with XYZ + RGB
        num_views: Number of random viewpoints to generate
        camera_distance_range: (min, max) distance from object center
        elevation_range: (min, max) height above object center
        azimuth_range: (min, max) azimuth angle in radians
        seed: Random seed for reproducibility

    Returns:
        Array of camera positions, shape (num_views, 3)
    """
    rng = np.random.default_rng(seed)

    obj_center = pointcloud[:, :3].mean(axis=0)

    cameras = []
    for _ in range(num_views):
        # Sample spherical coordinates
        azimuth = rng.uniform(*azimuth_range)
        distance = rng.uniform(*camera_distance_range)
        elevation = rng.uniform(*elevation_range)

        # Convert to Cartesian
        x = distance * np.cos(azimuth)
        y = distance * np.sin(azimuth)
        z = elevation

        cam_pos = obj_center + np.array([x, y, z])
        cameras.append(cam_pos)

    return np.array(cameras)


def compute_single_view_camera(
    pointcloud: np.ndarray,
    view_type: str = "front",
    camera_distance: float = 0.5,
    elevation: float = 0.2,
) -> np.ndarray:
    """Compute a single camera at a canonical viewpoint.

    Convenience function for single-camera setups.

    Args:
        pointcloud: Object point cloud (N, 6) with XYZ + RGB
        view_type: One of "front", "side", "overhead", "front_elevated"
        camera_distance: Distance from object center
        elevation: Height above object center (for non-overhead views)

    Returns:
        Array of camera positions, shape (1, 3)
    """
    obj_center = pointcloud[:, :3].mean(axis=0)

    if view_type == "front":
        cam_pos = obj_center + np.array([0.0, -camera_distance, elevation])
    elif view_type == "side":
        cam_pos = obj_center + np.array([-camera_distance, 0.0, elevation])
    elif view_type == "overhead":
        cam_pos = obj_center + np.array([0.0, 0.0, camera_distance])
    elif view_type == "front_elevated":
        # 45 degree angle from above, looking from front
        d = camera_distance / np.sqrt(2)
        cam_pos = obj_center + np.array([0.0, -d, d])
    else:
        raise ValueError(
            f"Unknown view_type: {view_type}. Use 'front', 'side', 'overhead', or 'front_elevated'"
        )

    return np.array([cam_pos])


def compute_eye_in_hand_camera(
    grasp_pose: np.ndarray,
    pointcloud: np.ndarray,
    offset: float = 0.1,
) -> np.ndarray:
    """Compute wrist-mounted (eye-in-hand) camera position.

    Simulates a camera mounted on the robot's wrist/end-effector,
    positioned behind the wrist and looking toward the grasp target.

    Args:
        grasp_pose: Shadow Hand pose (28,) where [0:3] is wrist translation
        pointcloud: Object point cloud (N, 6) - used for grasp direction
        offset: Distance behind wrist for camera placement

    Returns:
        Array of camera positions, shape (1, 3)
    """
    wrist_pos = grasp_pose[:3]
    obj_center = pointcloud[:, :3].mean(axis=0)

    # Grasp direction: from wrist toward object
    grasp_dir = obj_center - wrist_pos
    grasp_dir_norm = np.linalg.norm(grasp_dir)
    if grasp_dir_norm < 1e-6:
        grasp_dir = np.array([0.0, 0.0, 1.0])
    else:
        grasp_dir = grasp_dir / grasp_dir_norm

    # Camera is behind the wrist
    cam_pos = wrist_pos - offset * grasp_dir

    return np.array([cam_pos])


def compute_ego_and_thirdperson_cameras(
    grasp_pose: np.ndarray,
    pointcloud: np.ndarray,
    ego_offset: float = 0.15,
    thirdperson_position: np.ndarray | None = None,
) -> np.ndarray:
    """Compute two camera positions: egocentric (behind wrist) and third-person.

    Camera 1 (Egocentric): Behind the wrist, looking at the object.
        - Simulates what a wrist-mounted camera would see.

    Camera 2 (Third-person): Fixed viewpoint viewing both object and hand.
        - Default: positioned above and to the side.

    Args:
        grasp_pose: Shadow Hand pose (28,) where [0:3] is wrist translation
        pointcloud: Object point cloud (N, 6) with XYZ + RGB
        ego_offset: Distance behind wrist for egocentric camera
        thirdperson_position: Fixed 3D position for third-person camera.
            If None, uses a default position above and to the side.

    Returns:
        Array of camera positions, shape (2, 3)
    """
    # Extract positions
    wrist_pos = grasp_pose[:3]
    obj_center = pointcloud[:, :3].mean(axis=0)

    # Compute grasp direction (from wrist toward object)
    grasp_dir = obj_center - wrist_pos
    grasp_dir_norm = np.linalg.norm(grasp_dir)
    if grasp_dir_norm < 1e-6:
        grasp_dir = np.array([0.0, 0.0, 1.0])
    else:
        grasp_dir = grasp_dir / grasp_dir_norm

    # Camera 1: Egocentric (behind wrist, looking at object)
    # Position is behind the wrist along the negative grasp direction
    ego_camera = wrist_pos - ego_offset * grasp_dir

    # Camera 2: Third-person (fixed viewpoint)
    if thirdperson_position is not None:
        thirdperson_camera = np.array(thirdperson_position)
    else:
        # Default: above and to the side of the scene
        # Compute scene center (midpoint between wrist and object)
        scene_center = (wrist_pos + obj_center) / 2

        # Place camera above (Z+) and to the side (perpendicular to grasp direction)
        # Find a perpendicular direction
        if abs(grasp_dir[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])

        side_dir = np.cross(grasp_dir, up)
        side_dir = side_dir / np.linalg.norm(side_dir)

        # Third-person: elevated, to the side, looking at scene center
        elevation = 0.3  # Height above scene
        side_offset = 0.3  # Lateral offset
        thirdperson_camera = scene_center + elevation * np.array([0, 0, 1]) + side_offset * side_dir

    return np.array([ego_camera, thirdperson_camera])


def simulate_partial_rgbd_observation(
    pointcloud: np.ndarray,
    grasp_pose: np.ndarray,
    num_views: int = 3,
    camera_radius: float = 0.5,
    seed: int | None = None,
    hpr_radius_factor: float = 100.0,
    camera_mode: str = "hemisphere",
    thirdperson_position: np.ndarray | None = None,
    add_noise: bool = False,
    depth_noise_std: float = 0.002,
    lateral_noise_std: float = 0.001,
    outlier_ratio: float = 0.01,
    # Additional parameters for new camera modes
    fixed_views: list[str] | None = None,
    camera_distance_range: tuple[float, float] | None = None,
    elevation_range: tuple[float, float] | None = None,
    single_view_type: str = "front",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Simulate partial RGB-D observation from a full point cloud.

    This function simulates what an RGB-D camera would see by:
    1. Placing virtual cameras based on camera_mode
    2. Using Hidden Point Removal to determine visibility
    3. Merging visible points from all viewpoints
    4. Optionally adding realistic sensor noise

    Args:
        pointcloud: Full point cloud (N, 6) with XYZ + RGB
        grasp_pose: Shadow Hand pose (28,) where [0:3] is wrist translation
        num_views: Number of camera viewpoints (for "hemisphere" and "random" modes)
        camera_radius: Distance of cameras from object center
        seed: Random seed for reproducibility
        hpr_radius_factor: Radius factor for HPR algorithm
        camera_mode: Camera placement strategy:
            - "hemisphere": Multiple cameras on hemisphere opposite to grasp
            - "ego_thirdperson": One egocentric (behind wrist) + one third-person
            - "fixed_multiview": Fixed cameras at canonical positions (RLBench/CALVIN style)
            - "random": Random viewpoints for data augmentation
            - "overhead": Single top-down camera
            - "front": Single frontal camera
            - "side": Single side-view camera
            - "front_elevated": Single elevated front camera (45 deg)
            - "eye_in_hand": Wrist-mounted camera only
        thirdperson_position: Fixed position for third-person camera (only for "ego_thirdperson" mode)
        add_noise: Whether to add sensor noise to visible points
        depth_noise_std: Standard deviation of depth noise (meters). Default 2mm.
        lateral_noise_std: Standard deviation of lateral noise (meters). Default 1mm.
        outlier_ratio: Fraction of points to corrupt as outliers. Default 1%.
        fixed_views: List of view names for "fixed_multiview" mode.
            Options: "front", "back", "left", "right", "front_left", "front_right",
            "overhead", "overhead_front", "overhead_left", "overhead_right".
            Default: ["front", "left", "overhead"]
        camera_distance_range: (min, max) distance for "random" mode. Default: (0.3, 0.6)
        elevation_range: (min, max) elevation for "random" mode. Default: (0.1, 0.5)
        single_view_type: View type for single-view modes ("front", "side", "overhead", "front_elevated")

    Returns:
        Tuple of:
            partial_pc: Visible points (M, 6)
            visibility_mask: Boolean mask (N,) indicating visible points
            stats: Dict with visibility statistics
    """
    # Compute camera positions based on mode
    if camera_mode == "hemisphere":
        cameras = compute_grasp_aware_cameras(
            grasp_pose, pointcloud, num_views, camera_radius, seed
        )
    elif camera_mode == "ego_thirdperson":
        cameras = compute_ego_and_thirdperson_cameras(
            grasp_pose,
            pointcloud,
            ego_offset=camera_radius,
            thirdperson_position=thirdperson_position,
        )
    elif camera_mode == "fixed_multiview":
        cameras = compute_fixed_multiview_cameras(
            pointcloud,
            camera_distance=camera_radius,
            elevation=camera_radius * 0.6,  # Reasonable default
            views=fixed_views,
        )
    elif camera_mode == "random":
        dist_range = camera_distance_range or (0.3, 0.6)
        elev_range = elevation_range or (0.1, 0.5)
        cameras = compute_random_viewpoint_cameras(
            pointcloud,
            num_views=num_views,
            camera_distance_range=dist_range,
            elevation_range=elev_range,
            seed=seed,
        )
    elif camera_mode in ("front", "side", "overhead", "front_elevated"):
        cameras = compute_single_view_camera(
            pointcloud,
            view_type=camera_mode,
            camera_distance=camera_radius,
            elevation=camera_radius * 0.4,
        )
    elif camera_mode == "eye_in_hand":
        cameras = compute_eye_in_hand_camera(
            grasp_pose,
            pointcloud,
            offset=camera_radius,
        )
    else:
        valid_modes = [
            "hemisphere",
            "ego_thirdperson",
            "fixed_multiview",
            "random",
            "front",
            "side",
            "overhead",
            "front_elevated",
            "eye_in_hand",
        ]
        raise ValueError(f"Unknown camera_mode: {camera_mode}. Valid modes: {valid_modes}")

    # Apply HPR from each camera, merge results
    combined_mask = np.zeros(len(pointcloud), dtype=bool)
    for cam_pos in cameras:
        mask = hidden_point_removal(pointcloud, cam_pos, hpr_radius_factor)
        combined_mask |= mask

    # Extract visible points
    partial_pc = pointcloud[combined_mask].copy()

    # Optionally add sensor noise
    if add_noise and len(partial_pc) > 0:
        # Use centroid of cameras as reference for depth-dependent noise
        camera_centroid = cameras.mean(axis=0)
        partial_pc = add_sensor_noise(
            partial_pc,
            camera_position=camera_centroid,
            depth_noise_std=depth_noise_std,
            lateral_noise_std=lateral_noise_std,
            outlier_ratio=outlier_ratio,
            seed=seed,
        )

    stats = {
        "original_points": len(pointcloud),
        "visible_points": len(partial_pc),
        "visibility_ratio": combined_mask.mean(),
        "num_views": num_views,
        "camera_radius": camera_radius,
        "noise_applied": add_noise,
        "depth_noise_std": depth_noise_std if add_noise else 0.0,
    }

    return partial_pc, combined_mask, stats


# =============================================================================
# Transform class for integration with data pipeline
# =============================================================================


@dataclass(frozen=True)
class SimulatePartialRGBDObservation:
    """Transform to simulate partial RGB-D observation from full point cloud.

    This transform applies viewpoint-based visibility filtering using HPR
    (Hidden Point Removal) algorithm to simulate what an RGB-D camera would see,
    with optional realistic sensor noise.

    Supported camera modes (following standard robotics/VLA setups):
        - "hemisphere": Multiple cameras on hemisphere opposite to grasp (grasp-aware)
        - "ego_thirdperson": Wrist-mounted + fixed third-person (common in real robot setups)
        - "fixed_multiview": Fixed cameras at canonical positions (RLBench, CALVIN, ManiSkill style)
        - "random": Random viewpoints for domain randomization / data augmentation
        - "front": Single frontal camera
        - "side": Single side-view camera
        - "overhead": Single top-down camera (tabletop manipulation)
        - "front_elevated": Single elevated front camera (45 deg angle)
        - "eye_in_hand": Wrist-mounted camera only

    Args:
        num_views: Number of camera viewpoints (for "hemisphere" and "random" modes)
        camera_radius: Distance of cameras from object center / ego offset
        camera_mode: Camera placement strategy (see above)
        hpr_radius_factor: Radius factor for HPR algorithm
        seed: Random seed for reproducibility (None = random each call)
        add_noise: Whether to add sensor noise (depth noise, outliers)
        depth_noise_std: Standard deviation of depth noise in meters (default: 2mm)
        lateral_noise_std: Standard deviation of lateral noise in meters (default: 1mm)
        outlier_ratio: Fraction of points to corrupt as outliers (default: 1%)
        fixed_views: List of view names for "fixed_multiview" mode.
            Options: "front", "back", "left", "right", "front_left", "front_right",
            "overhead", "overhead_front", "overhead_left", "overhead_right".
        camera_distance_range: (min, max) distance for "random" mode
        elevation_range: (min, max) elevation for "random" mode

    Example:
        >>> # Standard multi-view setup (RLBench style)
        >>> transform = SimulatePartialRGBDObservation(
        ...     camera_mode="fixed_multiview",
        ...     fixed_views=["front", "left", "overhead"],
        ...     camera_radius=0.5,
        ... )

        >>> # Random viewpoint augmentation
        >>> transform = SimulatePartialRGBDObservation(
        ...     camera_mode="random",
        ...     num_views=3,
        ...     camera_distance_range=(0.3, 0.6),
        ...     add_noise=True,
        ... )

        >>> # Single overhead camera (tabletop)
        >>> transform = SimulatePartialRGBDObservation(
        ...     camera_mode="overhead",
        ...     camera_radius=0.4,
        ... )
    """

    num_views: int = 3
    camera_radius: float = 0.3
    camera_mode: str = "ego_thirdperson"
    hpr_radius_factor: float = 100.0
    seed: int | None = None
    # Sensor noise parameters
    add_noise: bool = False
    depth_noise_std: float = 0.002  # 2mm typical for consumer RGB-D
    lateral_noise_std: float = 0.001  # 1mm
    outlier_ratio: float = 0.01  # 1% outliers
    # Parameters for new camera modes
    fixed_views: tuple[str, ...] | None = None  # For fixed_multiview mode
    camera_distance_range: tuple[float, float] | None = None  # For random mode
    elevation_range: tuple[float, float] | None = None  # For random mode

    def __call__(self, data: DataDict) -> DataDict:
        """Apply partial observation simulation to point cloud.

        Expects data dict with:
            - "pointcloud": (N, 6) array with XYZ + RGB
            - "actions": (28,) array with grasp pose (first 3 = wrist position)

        Modifies data dict to have:
            - "pointcloud": (M, 6) partial observation (with optional noise)
            - "pointcloud_full": (N, 6) original point cloud (preserved)
            - "visibility_mask": (N,) boolean mask
            - "partial_obs_stats": dict with visibility statistics
        """
        pointcloud = data["pointcloud"]
        grasp_pose = data["actions"]

        # Convert tuple to list for fixed_views if needed
        fixed_views_list = list(self.fixed_views) if self.fixed_views else None

        # Simulate partial observation
        partial_pc, mask, stats = simulate_partial_rgbd_observation(
            pointcloud=pointcloud,
            grasp_pose=grasp_pose,
            num_views=self.num_views,
            camera_radius=self.camera_radius,
            seed=self.seed,
            hpr_radius_factor=self.hpr_radius_factor,
            camera_mode=self.camera_mode,
            add_noise=self.add_noise,
            depth_noise_std=self.depth_noise_std,
            lateral_noise_std=self.lateral_noise_std,
            outlier_ratio=self.outlier_ratio,
            fixed_views=fixed_views_list,
            camera_distance_range=self.camera_distance_range,
            elevation_range=self.elevation_range,
        )

        # Update data dict
        data["pointcloud_full"] = pointcloud  # Preserve original
        data["pointcloud"] = partial_pc  # Replace with partial
        data["visibility_mask"] = mask
        data["partial_obs_stats"] = stats

        return data

    def __str__(self) -> str:
        noise_str = f", noise={self.depth_noise_std * 1000:.1f}mm" if self.add_noise else ""
        if self.camera_mode == "fixed_multiview" and self.fixed_views:
            views_str = f", views={list(self.fixed_views)}"
        elif self.camera_mode in ("hemisphere", "random"):
            views_str = f", num_views={self.num_views}"
        else:
            views_str = ""
        return (
            f"SimulatePartialRGBDObservation("
            f"mode={self.camera_mode}"
            f"{views_str}, "
            f"radius={self.camera_radius}{noise_str})"
        )
