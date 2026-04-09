import numpy as np


class IBVSController:
    def __init__(self, camera_intrinsics, lambda_gain=0.5, Z_est=0.5, damping=1e-4):
        """
        camera_intrinsics: dict with fx, fy, cx, cy
        lambda_gain: proportional gain (λ)
        Z_est: estimated depth of target from camera (metres)
        damping: damping coefficient μ for DLS pseudo-inverse
        """
        self.fx = camera_intrinsics["fx"]
        self.fy = camera_intrinsics["fy"]
        self.cx = camera_intrinsics["cx"]
        self.cy = camera_intrinsics["cy"]
        self.lam = lambda_gain
        self.Z = Z_est
        self.mu = damping

    def _interaction_matrix(self, u, v):
        """
        Build 2×6 image interaction matrix Ls for a single point feature.

        Standard derivation (Chaumette & Hutchinson):
          x = (u - cx) / fx   — normalised image x (right → positive)
          y = (v - cy) / fy   — normalised image y (down  → positive)

          Ls = [[-1/Z,  0,   x/Z,  x*y,      -(1+x²),  y ],
                [  0, -1/Z,  y/Z,  1+y²,     -x*y,    -x ]]

        This matrix is expressed in the IBVS camera frame where:
          X = image-right direction,  Y = image-down direction,  Z = forward (optical axis).

        The camera-to-world rotation is handled externally in compute_joint_velocity()
        via the T_wc argument from Simulation.get_camera_frame().
        """
        x = (u - self.cx) / self.fx   # normalised x (rightward)
        y = (v - self.cy) / self.fy   # normalised y (downward)
        Z = self.Z

        Ls = np.array([
            [-1/Z,    0,   x/Z,   x*y,        -(1 + x**2),  y],
            [   0, -1/Z,   y/Z,   1 + y**2,   -x*y,        -x],
        ])
        return Ls

    def _dls_pinv(self, A):
        """Damped Least Squares pseudo-inverse: A^T (A A^T + μ²I)^{-1}"""
        m = A.shape[0]
        return A.T @ np.linalg.inv(A @ A.T + self.mu**2 * np.eye(m))

    def compute_camera_velocity(self, s, s_star):
        """
        Compute desired camera velocity v_c from pixel error.

        s, s_star : [u, v] arrays (pixel coordinates)
        Returns v_c (6,) — [vx, vy, vz, wx, wy, wz] in the IBVS camera frame.
        """
        e = s - s_star                              # (2,)
        Ls = self._interaction_matrix(s[0], s[1])  # (2, 6)
        Ls_pinv = self._dls_pinv(Ls)               # (6, 2)
        v_c = -self.lam * Ls_pinv @ e              # (6,) in camera frame
        return v_c

    def compute_joint_velocity(self, v_c, J, T_wc=None):
        """
        Map camera-frame velocity v_c to joint velocities via Jacobian pseudo-inverse.

        Parameters
        ----------
        v_c   : (6,) camera-frame velocity from compute_camera_velocity()
        J     : (6, 6) Jacobian in **world** frame (from Simulation.get_jacobian())
        T_wc  : (6, 6) camera-to-world velocity transform (from Simulation.get_camera_frame()).
                If None the identity is used (assumes J is already in camera frame).

        Returns
        -------
        q_dot : (6,) joint velocities
        """
        # Convert v_c from camera frame to world frame so it matches J's frame.
        if T_wc is not None:
            v_world = T_wc @ v_c        # (6,) in world frame
        else:
            v_world = v_c

        J_pinv = self._dls_pinv(J)     # (6, 6): J^T (J J^T + μ²I)^{-1}
        q_dot = J_pinv @ v_world       # (6,)
        return q_dot

    def is_converged(self, s, s_star, threshold=5.0):
        """Return True when pixel error norm is below threshold (pixels)."""
        return float(np.linalg.norm(s - s_star)) < threshold
