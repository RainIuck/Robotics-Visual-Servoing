import pybullet as p
import pybullet_data
import numpy as np
import os
import math

# Camera intrinsics
IMG_W = 640
IMG_H = 480
FOV = 60.0
NEAR = 0.01
FAR = 5.0
FX = IMG_W / (2 * math.tan(math.radians(FOV / 2)))
FY = FX
CX = IMG_W / 2.0
CY = IMG_H / 2.0
TARGET_HALF_EXTENTS = np.array([0.025, 0.025, 0.025])

CAMERA_INTRINSICS = {"fx": FX, "fy": FY, "cx": CX, "cy": CY,
                     "width": IMG_W, "height": IMG_H}

# UR5 arm joint names in order
UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Initial joint configuration:
# ee above workspace centre (~0.5, 0, 0.4), camera pointing straight down
# Verified: cam local-Z in world = [0, 0, -1]
UR5_HOME = [-0.22, -1.106, 1.072, 1.605, 1.571, -0.22]

# XY position of end-effector at home (used to seed target placement)
HOME_EE_XY = [0.5, 0.0]


class Simulation:
    def __init__(self, gui=True):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot_id = None
        self.target_id = None
        self.arm_joint_indices = []
        self.gripper_joint_index = None
        self.camera_link_index = None
        self.ee_link_index = None
        self.gripper_tip_link_index = None
        self.grasp_constraint_id = None

        self._load_robot()
        self._load_target()

    def _load_robot(self):
        urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "ur5_robotiq_85.urdf")
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        # Map joint names to indices
        num_joints = p.getNumJoints(self.robot_id)
        name_to_idx = {}
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            name_to_idx[joint_name] = i
            if link_name == "camera_link":
                self.camera_link_index = i
            if link_name == "ee_link":
                self.ee_link_index = i
            if link_name == "gripper_tip_link":
                self.gripper_tip_link_index = i

        self.arm_joint_indices = [name_to_idx[n] for n in UR5_JOINT_NAMES]
        self.gripper_joint_index = name_to_idx.get("finger_joint")

        # Move to home position
        for idx, angle in zip(self.arm_joint_indices, UR5_HOME):
            p.resetJointState(self.robot_id, idx, angle)

        # Let simulation settle
        for _ in range(100):
            p.stepSimulation()

    def _load_target(self, position=None):
        if position is None:
            # Place target randomly around the home ee XY position so it
            # starts inside the camera field of view
            x = HOME_EE_XY[0] + np.random.uniform(-0.15, 0.15)
            y = HOME_EE_XY[1] + np.random.uniform(-0.15, 0.15)
            z = 0.025  # resting on ground plane
            position = [x, y, z]

        # Create a small red cube as target
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=TARGET_HALF_EXTENTS.tolist())
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=TARGET_HALF_EXTENTS.tolist(),
                                     rgbaColor=[1, 0, 0, 1])
        self.target_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
        )
        return position

    def get_camera_image(self):
        if self.camera_link_index is None:
            raise RuntimeError("camera_link not found in URDF")

        link_state = p.getLinkState(self.robot_id, self.camera_link_index,
                                    computeForwardKinematics=True)
        cam_pos = link_state[4]   # world position
        cam_orn = link_state[5]   # world orientation (quaternion)

        # Camera optical axis = local Z in world frame
        rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot_mat[:, 2]   # local Z → optical axis

        # Choose an up vector that is never parallel to forward.
        # When camera points straight down ([0,0,-1]), world-X is a safe up.
        world_up = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.9:
            world_up = np.array([0.0, 1.0, 0.0])
        # Re-orthogonalise: right = forward × up, then up = right × forward
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        target_pos = np.array(cam_pos) + forward

        view_matrix = p.computeViewMatrix(cam_pos, target_pos.tolist(), up.tolist())
        proj_matrix = p.computeProjectionMatrixFOV(FOV, IMG_W / IMG_H, NEAR, FAR)

        _, _, rgb, _, _ = p.getCameraImage(
            IMG_W, IMG_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        # rgba -> rgb numpy array
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        return rgb_array

    def get_jacobian(self):
        # calculateJacobian requires vectors of length = numDof (all non-fixed joints)
        # Build full-length position vector from current joint states
        num_joints = p.getNumJoints(self.robot_id)
        q_full, qd_full, qdd_full = [], [], []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                q_full.append(p.getJointState(self.robot_id, i)[0])
                qd_full.append(0.0)
                qdd_full.append(0.0)

        jac_lin, jac_rot = p.calculateJacobian(
            self.robot_id,
            self.camera_link_index,
            [0.0, 0.0, 0.0],   # local point on camera_link
            q_full,
            qd_full,
            qdd_full,
        )
        # jac_lin/jac_rot are (3, numDof=12); keep only the 6 arm-joint columns
        jac_lin = np.array(jac_lin)[:, :6]   # (3, 6)
        jac_rot = np.array(jac_rot)[:, :6]   # (3, 6)
        J = np.vstack([jac_lin, jac_rot])     # (6, 6) — velocity in WORLD frame
        return J

    def get_camera_frame(self):
        """
        Return a 6×6 matrix T_wc that maps camera-frame velocity to world-frame velocity.

        PyBullet's calculateJacobian gives camera velocity expressed in the **world** frame,
        but the IBVS interaction matrix is derived in the **camera** frame
        (X = image-right, Y = image-down, Z = optical axis / forward).

        Use T_wc to convert before applying the Jacobian pseudo-inverse:
            v_world = T_wc @ v_cam
            q_dot   = J_world_pinv @ v_world
        """
        link_state = p.getLinkState(self.robot_id, self.camera_link_index,
                                    computeForwardKinematics=True)
        cam_orn = link_state[5]
        rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)

        # Reconstruct the same rendering frame axes used in get_camera_image()
        forward = rot_mat[:, 2]          # optical axis (cam local-Z in world)
        world_up = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.9:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # IBVS camera frame: X = right (image u↑), Y = -up (image v↑), Z = forward
        # R_wc maps camera-frame vectors to world frame (columns = cam axes in world)
        R_wc = np.column_stack([right, -up, forward])   # (3, 3)

        T_wc = np.zeros((6, 6))
        T_wc[:3, :3] = R_wc   # linear velocity transform
        T_wc[3:, 3:] = R_wc   # angular velocity transform
        return T_wc

    def get_target_position(self):
        """Return target world position."""
        if self.target_id is None:
            raise RuntimeError("Target not loaded")
        pos, _ = p.getBasePositionAndOrientation(self.target_id)
        return np.array(pos)

    def get_gripper_tip_pose(self):
        """Return gripper tip world position and orientation."""
        if self.gripper_tip_link_index is None:
            raise RuntimeError("gripper_tip_link not found in URDF")
        link_state = p.getLinkState(self.robot_id, self.gripper_tip_link_index,
                                    computeForwardKinematics=True)
        pos = np.array(link_state[4])
        orn = np.array(link_state[5])
        return pos, orn

    def move_gripper_tip_linear(self, target_pos, steps=240, max_vel=0.25):
        """Move gripper tip in a straight line using IK."""
        target_pos = np.array(target_pos, dtype=float)
        current_pos, current_orn = self.get_gripper_tip_pose()

        for alpha in np.linspace(0.0, 1.0, steps):
            interp_pos = (1.0 - alpha) * current_pos + alpha * target_pos
            joint_targets = p.calculateInverseKinematics(
                self.robot_id,
                self.gripper_tip_link_index,
                interp_pos.tolist(),
                targetOrientation=current_orn.tolist(),
                maxNumIterations=100,
                residualThreshold=1e-4,
            )
            self._set_arm_joint_positions(joint_targets[:len(self.arm_joint_indices)], max_vel=max_vel)
            self.step()

        self.hold_current_pose()

    def _set_arm_joint_positions(self, joint_targets, max_vel=0.25):
        for idx, target in zip(self.arm_joint_indices, joint_targets):
            p.setJointMotorControl2(
                self.robot_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target),
                maxVelocity=float(max_vel),
                force=300.0,
            )

    def _release_grasp_constraint(self):
        if self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = None

    def _attach_target_to_gripper(self):
        """Rigidly attach target to the gripper tip after successful closure."""
        self._release_grasp_constraint()
        target_pos = self.get_target_position()
        tip_pos, tip_orn = self.get_gripper_tip_pose()
        parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
            p.invertTransform(tip_pos.tolist(), tip_orn.tolist())[0],
            p.invertTransform(tip_pos.tolist(), tip_orn.tolist())[1],
            target_pos.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )
        self.grasp_constraint_id = p.createConstraint(
            self.robot_id,
            self.gripper_tip_link_index,
            self.target_id,
            -1,
            p.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            parent_frame_pos,
            [0.0, 0.0, 0.0],
            parentFrameOrientation=parent_frame_orn,
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        )

    def hold_current_pose(self):
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        joint_positions = [state[0] for state in joint_states]
        self._set_arm_joint_positions(joint_positions, max_vel=0.1)

    def lift_after_grasp(self, distance=0.12, steps=240, max_vel=0.25):
        """Lift gripper tip upward in world Z after grasping."""
        current_pos, _ = self.get_gripper_tip_pose()
        target_pos = current_pos + np.array([0.0, 0.0, distance])
        self.move_gripper_tip_linear(target_pos, steps=steps, max_vel=max_vel)

    def get_pregrasp_and_grasp_positions(self, clearance=0.14, descend_offset=0.055):
        """Compute pregrasp and grasp positions above the target."""
        target_pos = self.get_target_position()
        pregrasp = target_pos + np.array([0.0, 0.0, clearance])
        grasp = target_pos + np.array([0.0, 0.0, descend_offset])
        return pregrasp, grasp

    def execute_grasp_sequence(self, settle_steps=60):
        """Descend, close gripper, and lift the target."""
        pregrasp, grasp = self.get_pregrasp_and_grasp_positions(clearance=0.12, descend_offset=0.035)
        print(f"[IBVS] Pre-grasp pose: {pregrasp}")
        print(f"[IBVS] Grasp pose:     {grasp}")

        self.move_gripper_tip_linear(pregrasp, steps=180, max_vel=0.3)
        self.move_gripper_tip_linear(grasp, steps=220, max_vel=0.12)

        self.close_gripper()
        for _ in range(settle_steps * 4):
            self.step()

        self._attach_target_to_gripper()
        self.lift_after_grasp(distance=0.16, steps=220, max_vel=0.18)
        for _ in range(settle_steps):
            self.step()

    def set_joint_velocities(self, q_dot, max_vel=1.0):
        q_dot = np.clip(q_dot, -max_vel, max_vel)
        for idx, vel in zip(self.arm_joint_indices, q_dot):
            p.setJointMotorControl2(
                self.robot_id, idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(vel),
                force=150.0,
            )

    def open_gripper(self):
        if self.gripper_joint_index is not None:
            p.setJointMotorControl2(
                self.robot_id, self.gripper_joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=50.0,
            )

    def close_gripper(self):
        if self.gripper_joint_index is not None:
            p.setJointMotorControl2(
                self.robot_id, self.gripper_joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.7,
                force=50.0,
            )

    def step(self):
        p.stepSimulation()

    def disconnect(self):
        p.disconnect(self.client)
