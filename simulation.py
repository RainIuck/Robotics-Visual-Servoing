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

# Initial joint configuration (arm pointing downward, ready to servo)
UR5_HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


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
            # Random position in reachable workspace
            x = np.random.uniform(0.3, 0.6)
            y = np.random.uniform(-0.3, 0.3)
            z = 0.025  # resting on ground plane
            position = [x, y, z]

        # Create a small red cube as target
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025],
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

        # Camera forward direction: local +X after rpy="0 pi/2 0" mount
        rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot_mat[:, 2]   # local Z in world frame = optical axis
        up = rot_mat[:, 1] * -1   # local -Y as up

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
        joint_positions = [p.getJointState(self.robot_id, i)[0]
                           for i in self.arm_joint_indices]
        zero_vec = [0.0] * len(self.arm_joint_indices)

        # Local position of camera in ee frame (from URDF camera_joint origin)
        local_pos = [0.0, 0.0, 0.1]

        jac_lin, jac_rot = p.calculateJacobian(
            self.robot_id,
            self.camera_link_index,
            local_pos,
            joint_positions,
            zero_vec,
            zero_vec,
        )
        J = np.vstack([np.array(jac_lin), np.array(jac_rot)])  # (6, 6)
        return J

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
