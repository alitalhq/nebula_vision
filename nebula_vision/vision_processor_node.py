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
from nebula_interfaces.msg import BalloonArray, RectangleArray, GimbalMode
from cv_bridge import CvBridge
import cv2

# Sadece color detector kullanılacak
from .detectors.balloon_detector import BalloonDetector

class VisionProcessorNode(Node):
    """
    /camera/image_raw, /camera/camera_info, /gimbal/mode    ->
    /vision/balloons, /vision/rectangles, /vision/image_processed
    """

    def __init__(self):
        super().__init__("vision_processor_node")

        # ---------- Parametreler ----------
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("skip_n_frames", 0)

        # Kırmızı balon HSV range
        self.declare_parameter("balloon_lower_hsv", [0, 100, 100])
        self.declare_parameter("balloon_upper_hsv", [10, 255, 255])

        self.declare_parameter("rectangle_lower_hsv", [100, 100, 100])
        self.declare_parameter("rectangle_upper_hsv", [130, 255, 255])

        self.bridge = CvBridge()
        self.frame_counter = 0
        self.gimbal_mode = GimbalMode.MODE_LASER #kod testleri bitince burayı MODE_SAFE yapmayı unutma!!!

        # CameraInfo cache
        self.have_caminfo = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.img_w = self.img_h = 0

        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.skip_n = int(self.get_parameter("skip_n_frames").value)

        # Dedektör
        b_lower = self._get_int_list_param("balloon_lower_hsv", [0, 100, 100])
        b_upper = self._get_int_list_param("balloon_upper_hsv", [10, 255, 255])
        r_lower = self._get_int_list_param("rectangle_lower_hsv", [100, 100, 100])
        r_upper = self._get_int_list_param("rectangle_upper_hsv", [130, 255, 255])
        self.b_detector = BalloonDetector(balloon_lower=b_lower, balloon_upper=b_upper, rectangle_lower=r_lower, rectangle_upper=r_upper)
        self.get_logger().info(f"BalloonDetector aktif")

        # ---------- QoS ----------
        image_qos = QoSProfile(depth=1)
        image_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        info_qos = QoSProfile(depth=1)
        info_qos.reliability = QoSReliabilityPolicy.RELIABLE
        info_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        target_qos = QoSProfile(depth=1)
        target_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        debug_qos = QoSProfile(depth=1)
        debug_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        debug_qos.lifespan = Duration(seconds=0.15)

        mode_qos = QoSProfile(depth=1)
        mode_qos.reliability = QoSReliabilityPolicy.RELIABLE
        mode_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL


        # ---------- Subscribers ----------
        self.state_sub = self.create_subscription(GimbalMode,"/gimbal/mode", self.mode_callback, mode_qos)
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, image_qos)
        self.info_sub = self.create_subscription(CameraInfo, "/camera/camera_info", self.info_callback, info_qos)

        # ---------- Publishers ----------
        self.balloon_pub = self.create_publisher(BalloonArray, "/vision/balloons", target_qos)
        self.rectangle_pub = self.create_publisher(RectangleArray, "/vision/rectangles", target_qos)
        self.processed_image_pub = self.create_publisher(Image, "/vision/image_processed", debug_qos)

        # Dinamik parametre callback
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info("Vision Node hazır.")

    # ---------- Yardımcı ----------
    def _get_int_list_param(self, name, default):
        try:
            return list(self.get_parameter(name).value)
        except:
            return default
        
    # ---------- GimbalMode ----------
    def mode_callback(self, msg: GimbalMode):
        valid_modes = (
            GimbalMode.MODE_SAFE,
            GimbalMode.MODE_SEARCH,
            GimbalMode.MODE_LASER
        )

        if msg.mode not in valid_modes:
            self.get_logger().warning(f"Geçersiz gimbal mode alındı {msg.mode}")
            return
        
        if msg.mode == self.gimbal_mode:
            return
        
        old_mode = self.gimbal_mode
        self.gimbal_mode = msg.mode

        self.get_logger().info(f"Gimbal mode değişti: {old_mode} -> {self.gimbal_mode} ({msg.mode_text})")

    # ---------- CameraInfo ----------
    def info_callback(self, msg: CameraInfo):
        self.img_w, self.img_h = msg.width, msg.height
        if len(msg.k) == 9:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
        self.have_caminfo = True

    # ---------- Image ----------
    def image_callback(self, msg: Image):

        debug_img = None

        if self.skip_n > 0 and (self.frame_counter % (self.skip_n + 1)) != 0:
            self.frame_counter += 1
            return
        self.frame_counter += 1

        if self.b_detector is None and self.r_detector is None:
            return
        
        if self.gimbal_mode not in (GimbalMode.MODE_LASER, GimbalMode.MODE_SEARCH):
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge hatası: {e}")
            return
        
        balloons, debug_img, rect_center = self.b_detector.detect(cv_image, msg.header)

        if self.gimbal_mode == GimbalMode.MODE_LASER:
            if balloons:
                b_tarr = BalloonArray()
                b_tarr.header = msg.header
                b_tarr.balloons = balloons
                self.balloon_pub.publish(b_tarr)
            
        elif self.gimbal_mode == GimbalMode.MODE_SEARCH:
            pass #sonra doldur

        # Debug görüntüsü
        if self.publish_overlay and debug_img is not None:
            try:
                if len(debug_img.shape) == 2:
                    out = self.bridge.cv2_to_imgmsg(debug_img, encoding="mono8")
                else:
                    out = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")

                out.header = msg.header
                self.processed_image_pub.publish(out)
            except Exception as e:
                self.get_logger().error(f"Overlay yayın hatası: {e}")
                
    # ---------- Dinamik Parametreler ----------
    def _on_param_update(self, params):
        for p in params:

            if p.name == "publish_overlay":
                self.publish_overlay = bool(p.value)

            if p.name == "skip_n_frames":
                try:
                    v = int(p.value)
                    if v < 0:
                        return SetParametersResult(successful=False, reason="skip_n_frames >= 0 olmalı")
                    self.skip_n = v
                except Exception:
                    return SetParametersResult(successful=False, reason="skip_n_frames tamsayı olmalı")

            if p.name in ("balloon_lower_hsv", "balloon_upper_hsv"):
                b_lower = self._get_int_list_param("balloon_lower_hsv", [0, 100, 100])
                b_upper = self._get_int_list_param("balloon_upper_hsv", [10, 255, 255])
                if len(b_lower) != 3 or len(b_upper) != 3:
                    return SetParametersResult(False, "balloon HSV 3 elemanlı olmalı")
                self.b_detector = BalloonDetector(color_lower=b_lower, color_upper=b_upper)

            if p.name in ("rectangle_lower_hsv", "rectangle_upper_hsv"):
                r_lower = self._get_int_list_param("rectangle_lower_hsv", [100, 100, 100])
                r_upper = self._get_int_list_param("rectangle_upper_hsv", [130, 255, 255])
                if len(r_lower) != 3 or len(r_upper) != 3:
                    return SetParametersResult(False, "rectangle HSV 3 elemanlı olmalı")
                self.r_detector = RectangleDetector(color_lower=r_lower, color_upper=r_upper)

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
