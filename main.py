import numpy as np
import cv2

from simulation import Simulation, CAMERA_INTRINSICS
from vision import VisionProcessor
from controller import IBVSController
from plotter import ErrorPlotter

# ── Tunable parameters ────────────────────────────────────────────────────────
LAMBDA       = 0.003  # IBVS proportional gain
              # v_c is in physical units (m/s, rad/s) while pixel errors are
              # O(100 px).  With fx≈554 and Z≈0.5 m the effective gain on joint
              # velocities is ~140×LAMBDA, so 0.003 keeps q_dot ≲ 0.5 rad/s.
Z_EST        = 0.5    # estimated target depth from camera (metres)
DAMPING      = 1e-4   # DLS damping coefficient
CONVERGE_THR = 5.0    # pixel error threshold to trigger grasp (pixels)
MAX_STEPS    = 2000   # safety limit
SIM_STEP_HZ  = 240    # PyBullet default timestep
SHOW_DEBUG   = True   # display OpenCV debug window
# ─────────────────────────────────────────────────────────────────────────────


def main():
    sim        = Simulation(gui=True)
    vision     = VisionProcessor(CAMERA_INTRINSICS)
    controller = IBVSController(CAMERA_INTRINSICS,
                                lambda_gain=LAMBDA,
                                Z_est=Z_EST,
                                damping=DAMPING)
    plotter    = ErrorPlotter()

    # Desired feature: image centre
    s_star = np.array([CAMERA_INTRINSICS["width"]  / 2.0,
                       CAMERA_INTRINSICS["height"] / 2.0])

    sim.open_gripper()

    print("[IBVS] Starting visual servoing loop ...")
    for step in range(MAX_STEPS):
        # ── Step 1: get camera image ──────────────────────────────────────────
        rgb = sim.get_camera_image()

        # ── Step 2: extract visual feature ───────────────────────────────────
        s = vision.extract_feature(rgb)
        if s is None:
            print(f"[IBVS] step {step:4d} | target not visible, holding ...")
            sim.set_joint_velocities(np.zeros(6))
            sim.step()
            continue

        e = s - s_star
        plotter.update(e)

        # ── Step 3: compute control ───────────────────────────────────────────
        J     = sim.get_jacobian()       # (6,6) world frame
        T_wc  = sim.get_camera_frame()  # (6,6) camera→world velocity transform
        v_c   = controller.compute_camera_velocity(s, s_star)
        q_dot = controller.compute_joint_velocity(v_c, J, T_wc)

        # ── Step 4: drive joints ──────────────────────────────────────────────
        sim.set_joint_velocities(q_dot)
        sim.step()

        # ── Debug display ─────────────────────────────────────────────────────
        if SHOW_DEBUG:
            debug_img = vision.draw_debug(rgb, s, s_star)
            cv2.putText(debug_img,
                        f"step={step}  ||e||={np.linalg.norm(e):.1f}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Eye-in-Hand Camera", debug_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[IBVS] User quit.")
                break

        if step % 50 == 0:
            print(f"[IBVS] step {step:4d} | ||e|| = {np.linalg.norm(e):.2f} px")

        # ── Convergence check ─────────────────────────────────────────────────
        if controller.is_converged(s, s_star, CONVERGE_THR):
            print(f"[IBVS] Converged at step {step}! ||e|| = {np.linalg.norm(e):.2f} px")
            sim.set_joint_velocities(np.zeros(6))
            for _ in range(30):
                sim.step()
            sim.execute_grasp_sequence()
            print("[IBVS] Grasp executed.")
            break
    else:
        print(f"[IBVS] Reached max steps ({MAX_STEPS}) without convergence.")
        sim.set_joint_velocities(np.zeros(6))

    if SHOW_DEBUG:
        cv2.destroyAllWindows()

    plotter.plot("error_convergence.png")
    sim.disconnect()


if __name__ == "__main__":
    main()
