import os

import mujoco
import mujoco.viewer
import numpy as np
import transforms3d.quaternions as tq
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ============================================================================
# Interpolation helpers (for MjHO hand control)
# ============================================================================


def _interpolate_pose(pose1: np.ndarray, pose2: np.ndarray, step: int) -> np.ndarray:
    """SLERP interpolation between two 7D poses (translation + quaternion)."""
    trans1, quat1 = pose1[:3], pose1[3:7]
    trans2, quat2 = pose2[:3], pose2[3:7]
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2], scalar_first=True))
    trans_interp = np.linspace(trans1, trans2, step + 1)[1:]
    quat_interp = slerp(np.linspace(0, 1, step + 1))[1:].as_quat(scalar_first=True)
    return np.concatenate([trans_interp, quat_interp], axis=1)


def _interpolate_qpos(qpos1: np.ndarray, qpos2: np.ndarray, step: int) -> np.ndarray:
    """Linear interpolation between two joint position vectors."""
    return np.linspace(qpos1, qpos2, step + 1)[1:]


# ============================================================================
# MjHO: MuJoCo Hand-Object simulation
# ============================================================================


class MjHO:
    """MuJoCo Hand-Object simulation for contact extraction and grasp evaluation.

    Builds a MuJoCo scene with a Shadow Hand and an object, providing methods for:
    - Contact point extraction between hand and object
    - Hand pose control with interpolation
    - External force application on the object
    """

    hand_prefix: str = "child-"

    def __init__(
        self,
        obj_path,
        obj_scale,
        obj_density,
        hand_xml_path,
        hand_mocap,
        exclude_table_contact,
        friction_coef,
        has_floor_z0,
        debug_render=False,
        debug_viewer=False,
    ):
        self.hand_mocap = hand_mocap
        self.spec = mujoco.MjSpec()
        # dexter/ root: src/dexter/utils/ -> 4 levels up to dexter/
        self.spec.meshdir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        if debug_render or debug_viewer:
            self.spec.add_texture(
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[1.0, 1.0, 1.0],
                rgb2=[1.0, 1.0, 1.0],
                width=512,
                height=512,
            )
            self.spec.worldbody.add_light(
                name="spotlight",
                pos=[0, -1, 2],
                castshadow=False,
            )
            self.spec.worldbody.add_camera(
                name="closeup", pos=[0.0, 0.5, 0.5], xyaxes=[-1, 0, 0, 0, -1, 1]
            )

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, has_floor_z0)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}")

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # For ctrl
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

        self.debug_viewer = None
        self.debug_render = None
        if debug_viewer:
            self.debug_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.debug_viewer.sync()

        if debug_render:
            self.debug_render = mujoco.Renderer(self.model, 480, 640)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.debug_images = []
        self._ext_force = None

    def _add_hand(self, xml_path, mocap_base):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            # Resolve mesh path: try relative to XML file first, then relative to project root
            path = os.path.normpath(
                os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
            )
            if not os.path.exists(path):
                path = os.path.normpath(os.path.join(self.spec.meshdir, child_spec.meshdir, m.file))
            m.file = path
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(child_spec.worldbody, self.hand_prefix, "")
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )

    def _add_object(self, obj_path, obj_scale, obj_density, has_floor_z0):
        if has_floor_z0:
            self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        parts_folder = os.path.join(obj_path, "meshes")
        for file in os.listdir(parts_folder):
            if not file.endswith(".obj") or "convex_piece" not in file:
                continue
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
            )

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self):
        return self.data.qpos[-7:]

    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        """Extract hand-object and hand-hand contact information.

        Args:
            hand_qpos: Hand pose (29D: translation + quaternion + joints)
            obj_pose: Object pose (7D: translation + quaternion)
            obj_margin: Contact detection margin (m)

        Returns:
            Tuple of (hand-object contacts, hand-hand contacts).
            Each contact is a dict with keys:
            contact_dist, contact_pos, contact_normal, body1_name, body2_name
        """
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        self.reset_pose_qpos(hand_qpos, obj_pose)

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            if (body1_id > world_id and body1_id < hand_id and body2_id == object_id) or (
                body2_id > world_id and body2_id < hand_id and body1_id == object_id
            ):
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            elif (
                body1_id > world_id
                and body1_id < hand_id
                and body2_id > world_id
                and body2_id < hand_id
            ):
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )

        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def set_ext_force_on_obj(self, ext_force):
        self.data.xfrc_applied[-1] = ext_force
        self._ext_force = ext_force

    def reset_pose_qpos(self, hand_qpos, obj_pose):
        self._ext_force = None
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def control_hand_with_interp(self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10):
        if self.hand_mocap:
            pose_interp = _interpolate_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = _interpolate_qpos(
            self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer
        )
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)

    def _add_ext_force_arrow(self):
        """Add a red arrow to the scene indicating the external force direction."""
        force_vec = self._ext_force[:3]
        force_norm = np.linalg.norm(force_vec)
        if force_norm == 0:
            return
        scene = self.debug_render.scene
        if scene.ngeom >= scene.maxgeom:
            return

        z = force_vec / force_norm
        up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.array([x, y, z]).T.flatten()

        obj_pos = self.data.xpos[-1].copy()
        arrow_start = obj_pos - z * 0.25

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([0.012, 0.012, 0.2]),
            arrow_start,
            mat,
            np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        scene.ngeom += 1

    def control_hand_step(self, step_inner):
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            self.debug_render.update_scene(self.data, "closeup", self.debug_options)
            if self._ext_force is not None:
                self._add_ext_force_arrow()
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            raise NotImplementedError

    def render_pose(self, hand_qpos, obj_pose):
        """Render a static pose and return the image as numpy array."""
        self.reset_pose_qpos(hand_qpos, obj_pose)
        self.debug_render.update_scene(self.data, "closeup", self.debug_options)
        return self.debug_render.render()


# ============================================================================
# RobotKinematics
# ============================================================================


class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]]
                face = self.mj_model.mesh_face[mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }

        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self, root_pose):
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = (v["vert"] @ geom_rot.T + geom_trans) @ root_rot.T + root_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm
