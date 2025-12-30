import cv2
import numpy as np
from nebula_interfaces.msg import Balloon

class BalloonDetector:

    def __init__(self, balloon_lower, balloon_upper, rectangle_lower, rectangle_upper):
        self.lower = np.array(balloon_lower, dtype=np.uint8)
        self.upper = np.array(balloon_upper, dtype=np.uint8)

        self.blue_lower = np.array(rectangle_lower, dtype=np.uint8)
        self.blue_upper = np.array(rectangle_upper, dtype=np.uint8)

    def _is_red(self):
        return self.lower[0] < 15 or self.lower[0] > 164


    def detect(self, image, header):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        H, W = image.shape[:2]
        debug = image.copy()

        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_rect = None # (x, y, w, h)
        rect_center = None # (u_norm, v_norm)

        if blue_contours:
            best_c = max(blue_contours, key=cv2.contourArea)
            if cv2.contourArea(best_c) > 500:
                target_rect = cv2.boundingRect(best_c)
                rx, ry, rw, rh = target_rect
                rect_center = (float((rx + rw/2)/W), float((ry + rh/2)/H))
                # Dikdörtgen çevresini yeşil çiz
                cv2.rectangle(debug, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        # --- 2. Kırmızı Balonları Bul ---
        if self._is_red():
            mask1 = cv2.inRange(hsv, np.array([0, self.lower[1], self.lower[2]]), np.array([14, self.upper[1], self.upper[2]]))
            mask2 = cv2.inRange(hsv, np.array([165, self.lower[1], self.lower[2]]), np.array([179, self.upper[1], self.upper[2]]))
            mask = cv2.bitwise_or(mask1, mask2)
            color_label = "red"
        else:
            mask = cv2.inRange(hsv, self.lower, self.upper)
            color_label = "unknown"

        # --- Temizlik ---
        mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        balloons = []
        for idx, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < 200: continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue 

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.65: continue 
            
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 6: continue

            confidence = np.clip(circularity, 0.0, 1.0) 

            # İçeride mi kontrolü
            is_inside = False
            if target_rect:
                rx, ry, rw, rh = target_rect
                if rx < x < rx + rw and ry < y < ry + rh:
                    is_inside = True

            if is_inside:
                b = Balloon()
                b.id = idx
                b.u_norm, b.v_norm = float(x / W), float(y / H)
                b.confidence = float(confidence) 
                balloons.append(b)
                # İçerideyse YEŞİL
                cv2.circle(debug, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            else:
                # Dışarıdaysa MAVİ
                cv2.circle(debug, (int(x), int(y)), int(radius), (255, 0, 0), 2)

        # Nişangah
        cv2.line(debug, (W//2-12, H//2), (W//2+12, H//2), (0, 255, 0), 2)
        cv2.line(debug, (W//2, H//2-12), (W//2, H//2+12), (0, 255, 0), 2)

        return balloons, debug, rect_center
