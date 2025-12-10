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

# Sadece color detector kullanılacak
from .detectors.color_detector import ColorDetector


class VisionProcessorNode(Node):
    """
    /camera/image_raw -> /vision/balloons, /vision/image_processed
    Bu node sadece kırmızı balon tespiti yapar.
    """

    def __init__(self):
        super().__init__("vision_processor_node")

        # ---------- Parametreler ----------
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("skip_n_frames", 0)

        # Kırmızı balon HSV range
        self.declare_parameter("color_lower_hsv", [0, 100, 100])
        self.declare_parameter("color_upper_hsv", [10, 255, 255])

        self.bridge = CvBridge()
        self.frame_counter = 0

        # CameraInfo cache
        self.have_caminfo = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.img_w = self.img_h = 0

        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.skip_n = int(self.get_parameter("skip_n_frames").value)

        # Dedektör
        lower = self._get_int_list_param("color_lower_hsv", [0, 100, 100])
        upper = self._get_int_list_param("color_upper_hsv", [10, 255, 255])
        self.detector = ColorDetector(color_lower=lower, color_upper=upper)
        self.get_logger().info(f"ColorDetector aktif: lower={lower}, upper={upper}")

        # ---------- QoS ----------
        image_qos = QoSProfile(depth=1)
        image_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        info_qos = QoSProfile(depth=1)
        info_qos.reliability = QoSReliabilityPolicy.RELIABLE
        info_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        targets_qos = QoSProfile(depth=10)
        targets_qos.reliability = QoSReliabilityPolicy.RELIABLE

        debug_qos = QoSProfile(depth=1)
        debug_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        debug_qos.lifespan = Duration(seconds=0.15)

        # ---------- Subscribers ----------
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, image_qos)
        self.info_sub = self.create_subscription(CameraInfo, "/camera/camera_info", self.info_callback, info_qos)

        # ---------- Publishers ----------
        self.target_pub = self.create_publisher(TargetArray, "/vision/balloons", targets_qos)
        self.processed_image_pub = self.create_publisher(Image, "/vision/image_processed", debug_qos)

        # Dinamik parametre callback
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info("Internal Vision Node hazır (sadece color detector).")

    # ---------- Yardımcı ----------
    def _get_int_list_param(self, name, default):
        try:
            return list(self.get_parameter(name).value)
        except:
            return default

    # ---------- CameraInfo ----------
    def info_callback(self, msg: CameraInfo):
        self.img_w, self.img_h = msg.width, msg.height
        if len(msg.k) == 9:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
        self.have_caminfo = True

    # ---------- Frame Callback ----------
    def image_callback(self, msg: Image):

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

        # Sadece color detector
        try:
            targets, debug_img = self.detector.detect(cv_image, msg.header)
        except Exception as e:
            self.get_logger().error(f"Color detector çalışma hatası: {e}")
            return

        # Target publish
        if targets:
            tarr = TargetArray()
            tarr.header = msg.header
            tarr.targets = targets
            self.target_pub.publish(tarr)

        # Debug görüntüsü
        if self.publish_overlay:
            try:
                if debug_img is not None:
                    if len(debug_img.shape) == 2:
                        out = self.bridge.cv2_to_imgmsg(debug_img, encoding="mono8")
                    else:
                        out = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                else:
                    dbg = cv_image.copy()
                    H, W = dbg.shape[:2]
                    for t in (targets or []):
                        u_px = int(getattr(t, "u_norm", 0.5) * W)
                        v_px = int(getattr(t, "v_norm", 0.5) * H)
                        cv2.circle(dbg, (u_px, v_px), 12, (0, 255, 0), 2)
                    out = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")

                out.header = msg.header
                self.processed_image_pub.publish(out)
            except Exception as e:
                self.get_logger().error(f"Overlay yayın hatası: {e}")

    # ---------- Dinamik Parametreler ----------
    def _on_param_update(self, params):
        for p in params:
            if p.name == "publish_overlay":
                self.publish_overlay = bool(p.value)
            elif p.name == "skip_n_frames":
                try:
                    v = int(p.value)
                    if v < 0:
                        return SetParametersResult(successful=False, reason="skip_n_frames >= 0 olmalı")
                    self.skip_n = v
                except Exception:
                    return SetParametersResult(successful=False, reason="skip_n_frames tamsayı olmalı")

            elif p.name in ("color_lower_hsv", "color_upper_hsv"):
                lower = self._get_int_list_param("color_lower_hsv", [0, 100, 100])
                upper = self._get_int_list_param("color_upper_hsv", [10, 255, 255])
                self.detector = ColorDetector(color_lower=lower, color_upper=upper)

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
