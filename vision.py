import cv2
import numpy as np

# HSV range for red target (red wraps around 0/180 in HSV)
RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 100  # pixels^2, ignore noise


class VisionProcessor:
    def __init__(self, camera_intrinsics):
        self.fx = camera_intrinsics["fx"]
        self.fy = camera_intrinsics["fy"]
        self.cx = camera_intrinsics["cx"]
        self.cy = camera_intrinsics["cy"]
        self.img_w = camera_intrinsics["width"]
        self.img_h = camera_intrinsics["height"]

    def extract_feature(self, rgb_image):
        """
        Extract 2D centroid of the red target in pixel coordinates.
        Returns [u, v] (float) or None if target not found.
        """
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        return np.array([u, v])

    def draw_debug(self, rgb_image, s, s_star):
        """Draw detected centroid and target on image for visualization."""
        vis = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if s is not None:
            cv2.circle(vis, (int(s[0]), int(s[1])), 8, (0, 255, 0), -1)
        cv2.circle(vis, (int(s_star[0]), int(s_star[1])), 8, (255, 0, 0), 2)
        return vis
