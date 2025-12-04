from pyzbar import pyzbar
from hss_interfaces.msg import Target

class QRDetector:
    """
    Görüntüdeki QR kodlarını tespit eder ve verilerini çözer.
    """
    def __init__(self):
        pass

    def detect(self, image, header):
        """
        Verilen görüntüde QR kodlarını arar.
        """
        barcodes = pyzbar.decode(image)
        targets = []
        
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            u_center, v_center = x + w / 2, y + h / 2
            target = Target(id=0, u_norm=float(u_center/image.shape[1]), v_norm=float(v_center/image.shape[0]), class_label="qr_code", qr_code_data=barcode.data.decode("utf-8"))
            targets.append(target)
            
        return targets, image # QR detector için ayrı bir maske üretmiyoruz