import cv2
import numpy as np
from nebula_interfaces.msg import Balloon

class BalloonDetector:

    def __init__(self, color_lower, color_upper):
        self.lower = np.array(color_lower, dtype=np.uint8)
        self.upper = np.array(color_upper, dtype=np.uint8)

    def _is_red(self):
        return self.lower[0] < 15 or self.lower[0] > 164


    def detect(self, image, header):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Renk maskesi ---
        if self._is_red():
            # kırmızı = iki aralık
            lower1 = np.array([0, self.lower[1], self.lower[2]])
            upper1 = np.array([14, self.upper[1], self.upper[2]])
            lower2 = np.array([165, self.lower[1], self.lower[2]])
            upper2 = np.array([179, self.upper[1], self.upper[2]])

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
            color_label = "red"
        else:
            mask = cv2.inRange(hsv, self.lower, self.upper)
            color_label = "unknown"

        # --- Temizlik ---
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W = image.shape[:2]
        img_area = H * W

        balloons = []
        debug = image.copy()

        idx = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.65:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 8:
                continue

            # --- Confidence ---
            confidence = min(1.0, (area / img_area) * 8.0)

            b = Balloon()
            b.id = idx
            b.u_norm = float(x / W)
            b.v_norm = float(y / H)
            b.confidence = float(confidence)
            b.color_label = color_label

            balloons.append(b)
            idx += 1

            # --- Debug çizimleri ---
            center = (int(x), int(y))
            cv2.circle(debug, center, int(radius), (0, 255, 0), 2)
            cv2.circle(debug, center, 3, (0, 255, 0), -1)

        # --- Nişangah (+) ---
        cx, cy = W // 2, H // 2
        size = 12
        cv2.line(debug, (cx - size, cy), (cx + size, cy), (0, 255, 0), 1)
        cv2.line(debug, (cx, cy - size), (cx, cy + size), (0, 255, 0), 1)

        return balloons, debug
