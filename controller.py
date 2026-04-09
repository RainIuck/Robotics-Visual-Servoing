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
        Build 2x6 image interaction matrix Ls for a single point feature.
        Inputs u, v are pixel coordinates.
        """
        # Normalised image coordinates
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
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
        s, s_star: [u, v] arrays
        Returns v_c (6,) — [vx, vy, vz, wx, wy, wz]
        """
        e = s - s_star                              # (2,)
        Ls = self._interaction_matrix(s[0], s[1])   # (2, 6)
        Ls_pinv = self._dls_pinv(Ls)                # (6, 2)
        v_c = -self.lam * Ls_pinv @ e               # (6,)
        return v_c

    def compute_joint_velocity(self, v_c, J):
        """
        Map camera velocity v_c to joint velocities via Jacobian pseudo-inverse.
        J: (6, 6) Jacobian matrix
        Returns q_dot (6,)
        """
        J_pinv = self._dls_pinv(J.T).T              # DLS pinv of 6x6 J
        q_dot = J_pinv @ v_c                        # (6,)
        return q_dot

    def is_converged(self, s, s_star, threshold=5.0):
        """Return True when pixel error norm is below threshold (pixels)."""
        return float(np.linalg.norm(s - s_star)) < threshold
