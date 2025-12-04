#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSLivelinessPolicy,
    DurabilityPolicy,
)
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, CameraInfo
from nebula_interfaces.msg import TargetArray
from cv_bridge import CvBridge
import cv2

# Dedektörler
from .detectors.color_detector import ColorDetector
from .detectors.qr_detector import QRDetector

# YOLO opsiyonel (dosyan varsa aktive olur)
try:
    from .detectors.yolo_detector import YoloDetector  # detect(bgr, header) -> (targets, debug_img)
    _HAS_YOLO = True
except Exception:
    YoloDetector = None
    _HAS_YOLO = False


class VisionProcessorNode(Node):
    """
    /camera/image_raw -> /vision/targets, /vision/image_processed
    Dinamik parametre: detector_type=color|qr|yolo; overlay yayını ve frame skipping.
    CameraInfo'a abone; fx,fy,cx,cy ve (W,H) cache'ler.
    """

    def __init__(self):
        super().__init__("vision_processor_node")

        # ---------- Parametreler ----------
        self.declare_parameter("detector_type", "color")
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("skip_n_frames", 0)

        self.declare_parameter("color_lower_hsv", [20, 100, 100])
        self.declare_parameter("color_upper_hsv", [30, 255, 255])

        self.declare_parameter("yolo.model_path", "yolo.pt")
        self.declare_parameter("yolo.conf_thres", 0.25)
        self.declare_parameter("yolo.iou_thres", 0.45)
        self.declare_parameter("yolo.classes", [])
        self.declare_parameter("yolo.device", "cpu")

        self.bridge = CvBridge()
        self.frame_counter = 0

        # CameraInfo cache
        self.have_caminfo = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.img_w = self.img_h = 0

        # Param değerleri
        self.detector_type = self.get_parameter("detector_type").get_parameter_value().string_value
        self.publish_overlay = bool(self.get_parameter("publish_overlay").get_parameter_value().bool_value)
        self.skip_n = int(self.get_parameter("skip_n_frames").get_parameter_value().integer_value)

        # Dedektörü kur
        self.detector = None
        self._make_detector(initial=True)

        # ---------- QoS ve I/O ----------
        image_qos = QoSProfile(depth=1)
        image_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        image_qos.history = QoSHistoryPolicy.KEEP_LAST

        info_qos = QoSProfile(depth=1)
        info_qos.reliability = QoSReliabilityPolicy.RELIABLE
        info_qos.history = QoSHistoryPolicy.KEEP_LAST
        # camera_driver TRANSIENT_LOCAL ise geçmiş CameraInfo alınır
        info_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        targets_qos = QoSProfile(depth=10)
        targets_qos.reliability = QoSReliabilityPolicy.RELIABLE
        targets_qos.history = QoSHistoryPolicy.KEEP_LAST

        debug_qos = QoSProfile(depth=1)
        debug_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        debug_qos.history = QoSHistoryPolicy.KEEP_LAST
        debug_qos.liveliness = QoSLivelinessPolicy.AUTOMATIC
        debug_qos.lifespan = Duration(seconds=0.15)  # 150ms ömür

        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, image_qos)
        self.info_sub = self.create_subscription(CameraInfo, "/camera/camera_info", self.info_callback, info_qos)

        self.target_pub = self.create_publisher(TargetArray, "/vision/targets", targets_qos)
        self.processed_image_pub = self.create_publisher(Image, "/vision/image_processed", debug_qos)

        # Dinamik parametre callback
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f"VisionProcessorNode hazır: detector={self.detector_type}, overlay={self.publish_overlay}, skip_n={self.skip_n}"
        )

    # ---------- Yardımcılar ----------
    def _get_int_list_param(self, name, default):
        try:
            arr = self.get_parameter(name).get_parameter_value().integer_array_value
            return list(arr) if arr else list(default)
        except Exception:
            return list(default)

    def _make_detector(self, initial=False):
        dtype = self.get_parameter("detector_type").get_parameter_value().string_value
        self.detector_type = dtype

        if dtype == "color":
            lower = self._get_int_list_param("color_lower_hsv", [20, 100, 100])
            upper = self._get_int_list_param("color_upper_hsv", [30, 255, 255])
            self.detector = ColorDetector(color_lower=lower, color_upper=upper)
            self.get_logger().info(f"ColorDetector aktif: lower={lower}, upper={upper}")

        elif dtype == "qr":
            self.detector = QRDetector()
            self.get_logger().info("QRDetector aktif.")

        elif dtype == "yolo":
            if not _HAS_YOLO:
                self.get_logger().error("YOLO seçildi ama YoloDetector import edilemedi (detectors/yolo_detector.py var mı?).")
                self.detector = None
                return
            model = self.get_parameter("yolo.model_path").get_parameter_value().string_value
            conf = float(self.get_parameter("yolo.conf_thres").get_parameter_value().double_value)
            iou = float(self.get_parameter("yolo.iou_thres").get_parameter_value().double_value)
            cls = self._get_int_list_param("yolo.classes", [])
            dev = self.get_parameter("yolo.device").get_parameter_value().string_value
            try:
                self.detector = YoloDetector(
                    model_path=model,
                    conf_thres=conf,
                    iou_thres=iou,
                    classes=cls,
                    device=dev,
                )
                self.get_logger().info(
                    f"YoloDetector aktif: model={model}, conf={conf}, iou={iou}, classes={cls or 'ALL'}, device={dev}"
                )
            except Exception as e:
                self.get_logger().error(f"YoloDetector başlatılamadı: {e}")
                self.detector = None
        else:
            self.get_logger().error(f"Geçersiz detector_type: {dtype}")
            self.detector = None

        if not initial:
            self.get_logger().info("Dedektör yeniden yapılandırıldı.")

    # ---------- Callbacks ----------
    def info_callback(self, msg: CameraInfo):
        self.img_w, self.img_h = int(msg.width), int(msg.height)
        # K: fx, 0, cx, 0, fy, cy, 0, 0, 1
        if len(msg.k) == 9:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
        self.have_caminfo = True

    def image_callback(self, msg: Image):
        # Frame skipping
        if self.skip_n > 0 and (self.frame_counter % (self.skip_n + 1)) != 0:
            self.frame_counter += 1
            return
        self.frame_counter += 1

        if self.detector is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge hatası: {e}")
            return

        # Dedektörü çalıştır
        try:
            targets, debug_img = self.detector.detect(cv_image, msg.header)
        except Exception as e:
            self.get_logger().error(f"Dedektör çalışma hatası: {e}")
            return

        # Target publish
        if targets:
            tarr = TargetArray()
            tarr.header = msg.header
            tarr.targets = targets
            self.target_pub.publish(tarr)

        # Debug/overlay publish
        if self.publish_overlay:
            try:
                if debug_img is not None:
                    # color dedektörde mask mono olabilir
                    if len(debug_img.shape) == 2:  # mono8
                        out = self.bridge.cv2_to_imgmsg(debug_img, encoding="mono8")
                    else:
                        out = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                else:
                    # basit overlay: hedef merkezlerine daire + opsiyonel etiket
                    dbg = cv_image.copy()
                    H, W = dbg.shape[:2]
                    for t in (targets or []):
                        u_px = int(getattr(t, "u_norm", 0.5) * W)
                        v_px = int(getattr(t, "v_norm", 0.5) * H)
                        cv2.circle(dbg, (u_px, v_px), 12, (0, 255, 0), 2)
                        label = getattr(t, "qr_code_data", "") or getattr(t, "label", "")
                        if label:
                            cv2.putText(
                                dbg,
                                str(label),
                                (u_px, max(0, v_px - 14)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                lineType=cv2.LINE_AA,
                            )
                    out = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")

                out.header = msg.header
                self.processed_image_pub.publish(out)
            except Exception as e:
                self.get_logger().error(f"Overlay yayın hatası: {e}")

    # ---------- Dinamik Parametreler ----------
    def _on_param_update(self, params):
        recreate = False
        for p in params:
            if p.name == "detector_type":
                recreate = True
            elif p.name in ("color_lower_hsv", "color_upper_hsv") and self.detector_type == "color":
                recreate = True
            elif p.name.startswith("yolo.") and self.detector_type == "yolo":
                recreate = True
            elif p.name == "publish_overlay":
                self.publish_overlay = bool(p.value)
            elif p.name == "skip_n_frames":
                try:
                    v = int(p.value)
                    if v < 0:
                        return SetParametersResult(successful=False, reason="skip_n_frames >= 0 olmalı")
                    self.skip_n = v
                except Exception:
                    return SetParametersResult(successful=False, reason="skip_n_frames tamsayı olmalı")

        if recreate:
            self._make_detector()

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()