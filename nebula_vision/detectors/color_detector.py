import cv2
import numpy as np
from nebula_interfaces.msg import Target

class ColorDetector:
    """
    Belirli bir renk aralığındaki nesneleri tespit eder.
    """
    def __init__(self, color_lower, color_upper):
        self.color_lower = np.array(color_lower, dtype="uint8")
        self.color_upper = np.array(color_upper, dtype="uint8")

    def detect(self, image, header):
        """
        Verilen görüntüde renk tespiti yapar ve hedefleri döndürür.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        
        # Gürültüyü azaltmak için morfolojik operasyonlar
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        targets = []
        if len(contours) > 0:
            # En büyük konturu al
            c = max(contours, key=cv2.contourArea)
            ((u, v), radius) = cv2.minEnclosingCircle(c)
            
            if radius > 10: # Çok küçük hedefleri filtrele
                target = Target(id=0, u_norm=float(u/image.shape[1]), v_norm=float(v/image.shape[0]), class_label="color_blob")
                targets.append(target)

        return targets, mask