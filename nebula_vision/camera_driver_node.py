#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Tuple

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from ament_index_python.packages import get_package_share_directory

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
try:
    from camera_calibration_parsers import readCalibration
    _HAS_CCP = True
except Exception:
    import yaml
    _HAS_CCP = False

class CameraDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_driver_node")

        self.declare_parameter("profiles_file", "")
        self.declare_parameter("profile", "")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("frame_rate", 30.0)
        self.declare_parameter("camera_info_url", "package://nebula_vision/calib/internal.yaml")
        self.declare_parameter("frame_id", "internal_camera_optical_frame") # kalibrasyondan sonra düzeltmeyi unutma
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("fourcc", "MJPG")

        #durum bayrakları
        self.has_valid_calib_ = False # Geçerli bir kalibrasyon dosyasının yüklenip yüklenmediği
        self._calib_mismatch_warned_ = False  # Kalibrasyon boyutu ile akış boyutu uyuşmazlığı uyarısının verilip verilmediği
        self._first_info_published_ = False    # İlk görüntüyle birlikte CameraInfo'nun hemen yayınlanıp yayınlanmadığı

        self._load_parameters()
        self._maybe_apply_profile()
        self._load_parameters()

        image_qos_profile = QoSProfile(depth=1)
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.lifespan = Duration(seconds=0.1)

        cam_info_qos_profile = QoSProfile(depth=1)
        cam_info_qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        cam_info_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        cam_info_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        #publisherlar
        self.image_publisher_ = self.create_publisher(Image, "camera/image_raw", image_qos_profile)
        self.cam_info_publisher_ = self.create_publisher(CameraInfo, "camera/camera_info", cam_info_qos_profile)

        self.bridge_ = CvBridge()
        self.camera_info_msg_ = CameraInfo()
        self._load_camera_info() #camera_info mesajı yüklendi

        self.camera_handle_ = self._initialize_camera() #kamera başlatılır
        if self.camera_handle_ is None:
            raise RuntimeError("Kamera başlatılamadı.")


        self.image_timer_ = None
        self._create_image_timer()

        self.cam_info_timer_ = self.create_timer(1.0, self._cam_info_timer_cb) 

        self.add_on_set_parameters_callback(self._parameters_cb)

        # Başlangıçta kamera durumunu loglar
        self.get_logger().info(
            f"Kamera hazır -> ID:{self.camera_id_} " # Kamera ID'si
            f"{self.image_width_}x{self.image_height_}@{self.frame_rate_} " # Çözünürlük ve FPS
            f"FOURCC:{self.fourcc_} frame_id:{self.frame_id_}" # FOURCC ve frame_id
        )


    def _get_profiles_file_and_name(self) -> Tuple[str, str]:
        """
        profiles_file ve profile parametrelerini döndürür.
        """
        profiles_file = self.get_parameter("profiles_file").get_parameter_value().string_value
        profile_name  = self.get_parameter("profile").get_parameter_value().string_value
        return profiles_file, profile_name


    def _resolve_camera_info_url(self, url: str, profiles_file: str) -> str:
        """
        camera_info_url değerini platformdan bağımsız ve taşınabilir hale getirir.
        url'yi normalize eder
        Dönen değer: 'file://<ABSOLUTE_PATH>'
        """
        if not url: # URL boşsa
            return "" # Boş döndür

        # 1) file:// doğrudan normalize et
        if url.startswith("file://"):
            abs_path = url.replace("file://", "", 1)
            abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(abs_path)))
            return f"file://{abs_path}"

        # 2) package://<pkg>/... (ROS paket yolu çözümü)
        if url.startswith("package://"):
            try:
                pkg_and_rel = url.replace("package://", "", 1)
                pkg, rel = pkg_and_rel.split("/", 1)
                pkg_share = get_package_share_directory(pkg)
                abs_path = os.path.join(pkg_share, rel)
                return f"file://{os.path.abspath(abs_path)}"
            except Exception as e:
                self.get_logger().error(f"package:// URL çözümlenemedi: {url} ({e})")
                return ""

        # 3) ${ENV} ve ~ genişlet
        expanded = os.path.expandvars(os.path.expanduser(url))

        # 4) Göreli yol ise, profiles_file konumuna göre çöz
        if not os.path.isabs(expanded) and profiles_file:
            base = Path(profiles_file).resolve().parent
            expanded = str((base / expanded).resolve())

        return f"file://{os.path.abspath(expanded)}" # Son mutlak yolu 'file://' formatında döndürür


    def _maybe_apply_profile(self) -> None:
        """
        profiles_file + profile parametreleri verilmişse, ilgili profil sözlüğünü
        ROS parametrelerine uygular.
        """
        profiles_file, profile_name = self._get_profiles_file_and_name() # Profil dosya yolu ve adını alır
        if not profiles_file or not profile_name:
            return  # Profil kullanılmıyorsa fonksiyondan çıkar

        try:
            if not os.path.exists(profiles_file):
                self.get_logger().error(f"profiles_file bulunamadı: {profiles_file}")
                return

            with open(profiles_file, "r") as f: # Profilleri içeren dosyayı açar
                data = yaml.safe_load(f) or {} # YAML içeriğini güvenli bir şekilde yükler

            profiles = data.get("profiles", {}) # YAML'daki 'profiles' anahtarını alır
            if profile_name not in profiles: # İstenen profil YAML'da yoksa
                self.get_logger().error(f"Profil bulunamadı: '{profile_name}' (dosya: {profiles_file})")
                return

            prof = profiles[profile_name] or {} # İlgili profilin parametrelerini alır
            supported_keys = { # Desteklenen ROS parametrelerinin kümesi
                "camera_id", "frame_rate", "camera_info_url", "frame_id",
                "image_width", "image_height", "fourcc"
            }
            # Sadece desteklenen anahtarları profilden çeker
            applied = {k: prof[k] for k in prof.keys() if k in supported_keys}

            if not applied: # Hiçbir desteklenen parametre yoksa
                self.get_logger().warning(f"Profil '{profile_name}' boş ya da desteklenen anahtar yok.")
                return

            # camera_info_url varsa çözümlenir ve normalize edilir
            if "camera_info_url" in applied and applied["camera_info_url"]:
                # URL çözümleme fonksiyonunu çağırır
                applied["camera_info_url"] = self._resolve_camera_info_url(applied["camera_info_url"], profiles_file)

            # Parametre sunucusuna yaz (bu, parametrelerin düğümün içinde de güncellenmesini sağlar)
            param_objs = [Parameter(k, value=v) for k, v in applied.items()] # Uygulanacak parametreleri Parameter objelerine çevirir
            results = self.set_parameters(param_objs) # Parametreleri set eder
            if not all(r.successful for r in results): # Bazı parametreler set edilemediyse
                self.get_logger().warning(f"Profil parametrelerinden bazıları set edilemedi: {applied}")

            self.get_logger().info(f"Profil uygulandı: {profile_name} ({profiles_file}) -> {applied}") # Başarılı log

        except Exception as e: # Profil yükleme sırasında herhangi bir hata oluşursa
            self.get_logger().error(f"Profil yükleme hatası: {e}")


    def _load_parameters(self) -> None:
        """
        ROS parametre sunucusundan değerleri çek, doğrula ve URL’yi normalize et.
        """
        self.camera_id_ = self.get_parameter("camera_id").get_parameter_value().integer_value
        self.frame_rate_ = float(self.get_parameter("frame_rate").get_parameter_value().double_value)
        self.camera_info_url_ = self.get_parameter("camera_info_url").get_parameter_value().string_value
        self.frame_id_ = self.get_parameter("frame_id").get_parameter_value().string_value
        self.image_width_ = int(self.get_parameter("image_width").get_parameter_value().integer_value)
        self.image_height_ = int(self.get_parameter("image_height").get_parameter_value().integer_value)
        self.fourcc_ = self.get_parameter("fourcc").get_parameter_value().string_value


        if not (1.0 <= self.frame_rate_ <= 120.0):
            self.get_logger().warning(
                f"frame_rate={self.frame_rate_} geçersiz. 60.0 olarak ayarlanıyor."
            )
            self.frame_rate_ = 60.0

        if self.image_width_ <= 0 or self.image_height_ <= 0:
            self.get_logger().warning(
                f"Geçersiz çözünürlük ({self.image_width_}x{self.image_height_}). 640x480 olarak ayarlanıyor."
            )
            self.image_width_, self.image_height_ = 1280, 720

        if len(self.fourcc_) < 4:
            self.get_logger().warning(f"FOURCC='{self.fourcc_}' geçersiz. 'MJPG' olarak ayarlanıyor.")
            self.fourcc_ = "MJPG"
        else:
            self.fourcc_ = self.fourcc_[:4]

        # camera_info_url’ü normalize et (profil/CLI'dan gelmiş olabilir)
        profiles_file, _ = self._get_profiles_file_and_name()
        if self.camera_info_url_: # URL boş değilse
            self.camera_info_url_ = self._resolve_camera_info_url(self.camera_info_url_, profiles_file)


    def _set_prop(self, cap: cv2.VideoCapture, prop: int, value, label: str) -> None:
        """
        OpenCV cap.set sonucunu kontrol edip logla.
        """
        if not cap.set(prop, value):
            self.get_logger().warning(f"Kamera '{label}' ayarı başarısız/desteklenmiyor: {value}")


    def _initialize_camera(self):
        """
        Kamerayı mevcut parametrelerle başlatır.
        Linux'ta V4L2 backend’i açık seçik isteyerek set(...) tutarlılığını artırır.
        """
        try:
            cap = cv2.VideoCapture(self.camera_id_, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.get_logger().error(f"Kamera açılamadı! ID: {self.camera_id_}")
                return None

            self._set_prop(cap, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc_), "FOURCC")
            self._set_prop(cap, cv2.CAP_PROP_FRAME_WIDTH, self.image_width_, "WIDTH")
            self._set_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT, self.image_height_, "HEIGHT")
            self._set_prop(cap, cv2.CAP_PROP_FPS, self.frame_rate_, "FPS")
            self._set_prop(cap, cv2.CAP_PROP_BUFFERSIZE, 1, "BUFFERSIZE")

            self.get_logger().info("Kamera başarıyla açıldı ve ayarlandı.")
            return cap
        except Exception as e:
            self.get_logger().fatal(f"Kamera başlatılırken kritik hata: {e}")
            return None


    def _load_camera_info(self):
        """
        CameraInfo mesajını doldurur
        """
        clean_url = self.camera_info_url_.replace("file://", "")
        try:
            if _HAS_CCP:
                ok, camera_name, camera_info_msg = readCalibration(clean_url)
                if ok:
                    self.camera_info_msg_ = camera_info_msg
                    self.has_valid_calib_ = True # Geçerli kalibrasyon bayrağını True yapar
                    self.get_logger().info(f"Kalibrasyon yüklendi: {camera_name}")
                    return
                else:
                    self.get_logger().warning(f"CCP okuma başarısız: {clean_url}, PyYAML ile denenecek.")

            with open(clean_url, "r") as f:
                data = yaml.safe_load(f) or {}
            ci = CameraInfo()
            ci.width  = int(data.get("image_width", 0) or 0)
            ci.height = int(data.get("image_height", 0) or 0)

            K = data.get("camera_matrix", {}).get("data")
            D = data.get("distortion_coefficients", {}).get("data")
            R = data.get("rectification_matrix", {}).get("data")
            P = data.get("projection_matrix", {}).get("data")

            if K: ci.k = list(map(float, K))
            if D: ci.d = list(map(float, D))
            if R: ci.r = list(map(float, R))
            if P: ci.p = list(map(float, P))
            ci.distortion_model = data.get("distortion_model", "")

            self.camera_info_msg_ = ci
            self.has_valid_calib_ = True # Geçerli kalibrasyon bayrağını True yapar
            self.get_logger().info(f"Kalibrasyon (fallback) yüklendi: {clean_url}")
        except Exception as e:
            self.get_logger().error(f"Kalibrasyon yüklenirken hata: {e}")
            self.camera_info_msg_ = CameraInfo()
            self.has_valid_calib_ = False


    def _create_image_timer(self) -> None:
        """
        FPS'e göre görüntü yakalama timer'ını oluştur.
        """
        if self.image_timer_ is not None:
            self.destroy_timer(self.image_timer_)
        period = 1.0 / self.frame_rate_
        self.image_timer_ = self.create_timer(period, self._image_timer_cb)
        self.get_logger().info(f"Görüntü zamanlayıcısı {self.frame_rate_:.1f} Hz.")


    def _image_timer_cb(self) -> None:
        """
        Image_Raw mesajının callback fonksiyonu.
        """
        ret, frame = self.camera_handle_.read()
        if not ret:
            self.get_logger().warning("Kameradan görüntü alınamadı.")
            return

        now = self.get_clock().now().to_msg()
        msg = self.bridge_.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = now
        msg.header.frame_id = self.frame_id_
        self.image_publisher_.publish(msg)

        # İlk görüntüyle birlikte CameraInfo’u bir defa hemen yayınlar (alıcılar için)
        if self.has_valid_calib_ and not self._first_info_published_:
            self.camera_info_msg_.header.stamp = now
            self.camera_info_msg_.header.frame_id = self.frame_id_
            self.cam_info_publisher_.publish(self.camera_info_msg_)
            self._first_info_published_ = True


    def _cam_info_timer_cb(self) -> None:
        """
        Camera_Info mesajının callback fonksiyonu.
        """
        if not self.has_valid_calib_:
            return

        if (
            self.camera_info_msg_.width
            and self.camera_info_msg_.height
            and (self.camera_info_msg_.width != self.image_width_
                 or self.camera_info_msg_.height != self.image_height_)
        ):
            if not self._calib_mismatch_warned_:
                self.get_logger().warning(
                    f"CameraInfo ({self.camera_info_msg_.width}x{self.camera_info_msg_.height}) "
                    f"≠ capture ({self.image_width_}x{self.image_height_}). "
                    f"YAML’ı akış çözünürlüğüne göre güncelleyin."
                )
                self._calib_mismatch_warned_ = True

        now = self.get_clock().now().to_msg()
        self.camera_info_msg_.header.stamp = now
        self.camera_info_msg_.header.frame_id = self.frame_id_
        self.cam_info_publisher_.publish(self.camera_info_msg_)


    def _parameters_cb(self, params: List[Parameter]) -> SetParametersResult:
        """
        Parametre değişimlerini güvenli uygula ve gerekli yeniden başlatmaları tetikle.
        """
        new_camera_id = self.camera_id_
        new_frame_rate = self.frame_rate_
        new_width = self.image_width_
        new_height = self.image_height_
        new_fourcc = self.fourcc_

        reinit_camera = False # Kamera yeniden başlatılacak mı? (ID, Çözünürlük, FOURCC değişirse)
        recreate_timer = False # Timer yeniden oluşturulacak mı? (FPS değişirse)
        rerun_profile = False # Profil yeniden uygulanacak mı? (profiles_file veya profile değişirse)

        for p in params: # Değiştirilen her bir parametre için döngü

            if p.name == "frame_rate": # FPS değişirse
                try:
                    v = float(p.value)
                    if not (1.0 <= v <= 120.0):
                        return SetParametersResult(successful=False, reason="frame_rate 1–120 aralığında olmalı")
                    new_frame_rate = v
                    recreate_timer = True
                except Exception:
                    return SetParametersResult(successful=False, reason="frame_rate sayısal olmalı")
                

            elif p.name in ("image_width", "image_height"): # Çözünürlük değişirse
                try:
                    iv = int(p.value)
                    if iv <= 0:
                        return SetParametersResult(successful=False, reason=f"{p.name} > 0 olmalı")
                    if p.name == "image_width":
                        new_width = iv
                    else:
                        new_height = iv
                    reinit_camera = True
                except Exception:
                    return SetParametersResult(successful=False, reason=f"{p.name} tamsayı olmalı")
                

            elif p.name == "camera_id": # Kamera ID'si değişirse
                try:
                    new_camera_id = int(p.value)
                    reinit_camera = True
                except Exception:
                    return SetParametersResult(successful=False, reason="camera_id tamsayı olmalı")


            elif p.name == "fourcc": # FOURCC değişirse
                s = str(p.value)
                if len(s) < 4:
                    return SetParametersResult(successful=False, reason="fourcc 4 karakter olmalı (örn. MJPG, YUYV)")
                new_fourcc = s[:4]
                reinit_camera = True


            elif p.name == "camera_info_url": # Kalibrasyon URL'si değişirse
                profiles_file, _ = self._get_profiles_file_and_name()
                self.camera_info_url_ = self._resolve_camera_info_url(str(p.value), profiles_file)
                self._load_camera_info()


            elif p.name == "frame_id": # Frame ID değişirse
                self.frame_id_ = str(p.value)


            elif p.name in ("profiles_file", "profile"): # Profil ayarları değişirse
                rerun_profile = True

        # Profil yeniden uygulanmak istenirse:
        if rerun_profile:
            self._maybe_apply_profile()
            self._load_parameters()
            reinit_camera = True
            recreate_timer = True

        # Kamera yeniden başlatma gerekirse
        if reinit_camera:
            if self.camera_handle_ is not None:
                self.camera_handle_.release()

            self.camera_id_, self.image_width_, self.image_height_, self.fourcc_ = (
                new_camera_id,
                new_width,
                new_height,
                new_fourcc,
            )
            self.camera_handle_ = self._initialize_camera()
            if self.camera_handle_ is None:
                self._load_parameters()
                self.camera_handle_ = self._initialize_camera()
                return SetParametersResult(successful=False, reason="Kamera yeni ayarlarla açılamadı")

        if recreate_timer:
            self.frame_rate_ = new_frame_rate
            self._create_image_timer()

        return SetParametersResult(successful=True)


    def destroy_node(self) -> None:
        """
        Düğüm kapanırken kaynakları serbest bırak.
        """
        self.get_logger().info("Kamera kapatılıyor.")
        if hasattr(self, "camera_handle_") and self.camera_handle_ is not None:
            self.camera_handle_.release()
        super().destroy_node()



def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CameraDriverNode()
        rclpy.spin(node)

    except (KeyboardInterrupt, RuntimeError) as e:
        if isinstance(e, RuntimeError):
            print(f"Düğüm başlatılamadı: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
