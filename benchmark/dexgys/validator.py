"""
Isaac Gym grasp validator.
Runs stability tests (6 rotations) per grasp and checks hand-object contact.
"""

import math
import os
from time import sleep

import numpy as np
from isaacgym import gymapi
from PIL import Image

gym = gymapi.acquire_gym()


class IsaacValidator:
    def __init__(
        self,
        mode="direct",
        hand_friction=3.0,
        obj_friction=3.0,
        threshold_dis=0.1,
        sim_step=100,
        gpu=0,
        debug_interval=0.05,
        save_render=False,
        render_dir="./rendering",
    ):

        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.threshold_dis = threshold_dis
        self.gpu = gpu
        self.sim_step = sim_step
        self.save_render = save_render
        self.render_dir = render_dir
        self.render_step = 0
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.joint_names = [
            "robot0:FFJ3",
            "robot0:FFJ2",
            "robot0:FFJ1",
            "robot0:FFJ0",
            "robot0:MFJ3",
            "robot0:MFJ2",
            "robot0:MFJ1",
            "robot0:MFJ0",
            "robot0:RFJ3",
            "robot0:RFJ2",
            "robot0:RFJ1",
            "robot0:RFJ0",
            "robot0:LFJ4",
            "robot0:LFJ3",
            "robot0:LFJ2",
            "robot0:LFJ1",
            "robot0:LFJ0",
            "robot0:THJ4",
            "robot0:THJ3",
            "robot0:THJ2",
            "robot0:THJ1",
            "robot0:THJ0",
        ]
        self.hand_asset = None
        self.obj_asset = None
        # joint name -> DOF index is identical for every env (same hand asset), but
        # gym.find_actor_dof_index does a string search. Resolve it once on the first
        # env and reuse, instead of 22 (or 44) string lookups per env.
        self._joint_indices = None

        self.sim_params = gymapi.SimParams()

        # set common parameters
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)

        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0

        self.sim_params.use_gpu_pipeline = False
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 800
        self.camera_props.height = 600
        self.camera_props.use_collision_geometry = True

        # set viewer
        self.viewer = None
        if mode == "gui" or save_render:
            self.has_viewer = True
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0, 0, 1), gymapi.Vec3(0, 0, 0))
        else:
            self.has_viewer = False

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.fix_base_link = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0), gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1 * math.pi)
            ),  # z axis 180
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * math.pi),
            ),  # z axis 90
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -0.5 * math.pi),
            ),  # z axis -90
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi),
            ),  # x axis 90
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi),
            ),  # x axis -90
        ]

    def _resolve_joint_indices(self, env, hand_actor_handle):
        """Map self.joint_names -> DOF indices once and cache for the validator's life.

        The mapping depends only on the hand asset (constant across envs and across
        reset_simulator), so resolving it once avoids a string search per joint per env.
        """
        if self._joint_indices is None:
            self._joint_indices = [
                gym.find_actor_dof_index(env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR)
                for joint in self.joint_names
            ]
        return self._joint_indices

    def set_asset(self, hand_root, hand_file, obj_root, obj_file):
        self.hand_asset = gym.load_asset(self.sim, hand_root, hand_file, self.hand_asset_options)
        self.obj_asset = gym.load_asset(self.sim, obj_root, obj_file, self.obj_asset_options)

    def add_env(self, hand_rotation, hand_translation, hand_qpos, obj_scale, target_qpos=None):
        for test_rot in self.test_rotations:
            env = gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 6)
            self.envs.append(env)
            pose = gymapi.Transform()
            pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])
            pose.p = gymapi.Vec3(*hand_translation)
            pose = test_rot * pose
            hand_actor_handle = gym.create_actor(env, self.hand_asset, pose, "shand", 0, -1)
            self.hand_handles.append(hand_actor_handle)
            hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
            hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            hand_props["stiffness"].fill(1000)
            hand_props["damping"].fill(0.0)
            gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)
            dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
            joint_indices = self._resolve_joint_indices(env, hand_actor_handle)
            for i, joint_idx in enumerate(joint_indices):
                dof_states["pos"][joint_idx] = hand_qpos[i]
            gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)
            if target_qpos is not None:
                for i, joint_idx in enumerate(joint_indices):
                    dof_states["pos"][joint_idx] = target_qpos[i]
            gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_states["pos"])

            hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
            hand_rigid_body_set = set()
            for i in range(gym.get_actor_rigid_body_count(env, hand_actor_handle)):
                hand_rigid_body_set.add(
                    gym.get_actor_rigid_body_index(env, hand_actor_handle, i, gymapi.DOMAIN_ENV)
                )
            self.hand_rigid_body_sets.append(hand_rigid_body_set)
            for i in range(len(hand_shape_props)):
                hand_shape_props[i].friction = self.hand_friction
            gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 0)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            pose = test_rot * pose
            obj_actor_handle = gym.create_actor(env, self.obj_asset, pose, "obj", 0, 1)
            self.obj_handles.append(obj_actor_handle)
            gym.set_actor_scale(env, obj_actor_handle, obj_scale)
            obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
            obj_rigid_body_set = set()
            for i in range(gym.get_actor_rigid_body_count(env, obj_actor_handle)):
                obj_rigid_body_set.add(
                    gym.get_actor_rigid_body_index(env, obj_actor_handle, i, gymapi.DOMAIN_ENV)
                )
            self.obj_rigid_body_sets.append(obj_rigid_body_set)
            for i in range(len(obj_shape_props)):
                obj_shape_props[i].friction = self.obj_friction
            gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)

            # Render frame after placing hand and object
            if self.save_render:
                self.render_setup_frame(env, len(self.envs) - 1)

            # Update viewer if available
            if self.has_viewer:
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

    def add_env_single(
        self, hand_rotation, hand_translation, hand_qpos, obj_scale, index=0, target_qpos=None
    ):
        test_rot = self.test_rotations[index]
        env = gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 6)
        self.envs.append(env)
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])
        pose.p = gymapi.Vec3(*hand_translation)
        pose = test_rot * pose
        hand_actor_handle = gym.create_actor(env, self.hand_asset, pose, "shand", 0, -1)
        self.hand_handles.append(hand_actor_handle)
        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
        hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        hand_props["stiffness"].fill(1000)
        hand_props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)
        dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
        joint_indices = self._resolve_joint_indices(env, hand_actor_handle)
        for i, joint_idx in enumerate(joint_indices):
            dof_states["pos"][joint_idx] = hand_qpos[i]
        gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)
        if target_qpos is not None:
            for i, joint_idx in enumerate(joint_indices):
                dof_states["pos"][joint_idx] = target_qpos[i]
        gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_states["pos"])

        hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
        hand_rigid_body_set = set()
        for i in range(gym.get_actor_rigid_body_count(env, hand_actor_handle)):
            hand_rigid_body_set.add(
                gym.get_actor_rigid_body_index(env, hand_actor_handle, i, gymapi.DOMAIN_ENV)
            )
        self.hand_rigid_body_sets.append(hand_rigid_body_set)
        for i in range(len(hand_shape_props)):
            hand_shape_props[i].friction = self.hand_friction
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        pose = test_rot * pose
        obj_actor_handle = gym.create_actor(env, self.obj_asset, pose, "obj", 0, 1)
        self.obj_handles.append(obj_actor_handle)
        gym.set_actor_scale(env, obj_actor_handle, obj_scale)
        obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
        obj_rigid_body_set = set()
        for i in range(gym.get_actor_rigid_body_count(env, obj_actor_handle)):
            obj_rigid_body_set.add(
                gym.get_actor_rigid_body_index(env, obj_actor_handle, i, gymapi.DOMAIN_ENV)
            )
        self.obj_rigid_body_sets.append(obj_rigid_body_set)
        for i in range(len(obj_shape_props)):
            obj_shape_props[i].friction = self.obj_friction
        gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)

        # Render frame after placing hand and object
        if self.save_render:
            self.render_setup_frame(env, len(self.envs) - 1)

        # Update viewer if available
        if self.has_viewer:
            gym.step_graphics(self.sim)
            gym.draw_viewer(self.viewer, self.sim, False)

    def run_sim(self):
        cam_handles = []
        if self.save_render and len(self.envs) > 0:
            for env in self.envs:
                cam_handle = gym.create_camera_sensor(env, self.camera_props)
                gym.set_camera_location(
                    cam_handle, env, gymapi.Vec3(0.6, 0.6, 0.8), gymapi.Vec3(0, 0, 0)
                )
                cam_handles.append(cam_handle)
        for step in range(self.sim_step):
            gym.simulate(self.sim)
            if self.has_viewer:
                sleep(self.debug_interval)
                if gym.query_viewer_has_closed(self.viewer):
                    break
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

            if self.save_render and step % 10 == 0:
                self.save_simulation_frame(step, cam_handles)

        success = []
        for i, env in enumerate(self.envs):
            contacts = gym.get_env_rigid_contacts(env)
            flag = False
            for contact in contacts:
                if (contact[2] in self.hand_rigid_body_sets[i]) and (
                    contact[3] in self.obj_rigid_body_sets[i]
                ):
                    flag = True
                    break
                if (contact[3] in self.hand_rigid_body_sets[i]) and (
                    contact[2] in self.obj_rigid_body_sets[i]
                ):
                    flag = True
                    break
            success.append(flag)

        # Save final result frames
        if self.save_render and cam_handles:
            self.save_final_result_frame(cam_handles, success)

        # Clean up camera sensors
        if self.save_render and cam_handles:
            for i, cam_handle in enumerate(cam_handles):
                gym.destroy_camera_sensor(self.sim, self.envs[i], cam_handle)

        if self.save_render:
            self.render_step += 1

        return success

    def reset_simulator(self):
        gym.destroy_sim(self.sim)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        if self.has_viewer:
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.hand_asset = None
        self.obj_asset = None

    def save_simulation_frame(self, step, cam_handles):
        if not self.save_render or not cam_handles:
            return

        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)
        for i, (env, cam_handle) in enumerate(zip(self.envs, cam_handles)):
            try:
                color_image = gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                color_image = color_image.reshape(
                    self.camera_props.height, self.camera_props.width, 4
                )
                rgb_image = color_image[:, :, :3]

                img = Image.fromarray(rgb_image.astype(np.uint8))
                filename = (
                    f"{self.render_dir}/sim_{self.render_step:05d}_env_{i:02d}_step_{step:03d}.png"
                )
                img.save(filename)

                if i == 0 and step == 0:
                    print(f"Started saving simulation frames to {self.render_dir}/")

            except Exception as e:
                print(f"Failed to save frame for env {i}, step {step}: {e}")

    def save_final_result_frame(self, cam_handles, success_results):
        if not self.save_render or not cam_handles:
            return

        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)

        for i, (env, cam_handle) in enumerate(zip(self.envs, cam_handles)):
            try:
                color_image = gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                color_image = color_image.reshape(
                    self.camera_props.height, self.camera_props.width, 4
                )
                rgb_image = color_image[:, :, :3]

                result_text = (
                    "SUCCESS" if i < len(success_results) and success_results[i] else "FAILED"
                )
                filename = (
                    f"{self.render_dir}/final_{self.render_step:05d}_env_{i:02d}_{result_text}.png"
                )

                img = Image.fromarray(rgb_image.astype(np.uint8))
                img.save(filename)

            except Exception as e:
                print(f"Failed to save final frame for env {i}: {e}")

    def render_setup_frame(self, env, env_index):
        if not self.save_render:
            return

        try:
            os.makedirs(self.render_dir, exist_ok=True)

            cam_handle = gym.create_camera_sensor(env, self.camera_props)
            gym.set_camera_location(
                cam_handle, env, gymapi.Vec3(0.6, 0.6, 0.8), gymapi.Vec3(0, 0, 0)
            )

            gym.step_graphics(self.sim)
            gym.render_all_camera_sensors(self.sim)

            color_image = gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
            color_image = color_image.reshape(self.camera_props.height, self.camera_props.width, 4)
            rgb_image = color_image[:, :, :3]

            filename = f"{self.render_dir}/setup_{self.render_step:05d}_env_{env_index:02d}.png"
            img = Image.fromarray(rgb_image.astype(np.uint8))
            img.save(filename)

            gym.destroy_camera_sensor(self.sim, env, cam_handle)

        except Exception as e:
            print(f"Failed to render setup frame for env {env_index}: {e}")

    def destroy(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.sim)
