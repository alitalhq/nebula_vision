#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 Camera Driver (profil destekli, dinamik parametre güncelleme, düşük gecikme)
- Tek bir profil YAML dosyasından (profiles_file + profile) seçili kameranın TÜM parametrelerini uygular
- OpenCV (V4L2) ile kamera yakalama
- image_raw ve camera_info yayınları (uygun QoS)
- YAML kalibrasyonunu camera_calibration_parsers ile yükleme
- Çalışma anında parametre güncelleme (FPS/çözünürlük/FourCC/kamera_id/...) ve yeniden başlatma
- Saha kullanımında sık görülen hatalara karşı korumalar ve bilgilendirici loglar
- camera_info_url çözümleyici: package://, ${ENV}, ~, göreli yol (profiles_file konumuna göre) ve file:// destekler
"""

import os # İşletim sistemi ile etkileşim (dosya yolları, çevre değişkenleri) için standart kütüphane
from pathlib import Path # Dosya yollarıyla daha modern ve nesne tabanlı çalışma için
from typing import List, Tuple # Tip ipuçları için (kodu okumayı kolaylaştırır)

import yaml  # YAML dosyalarını okumak/yazmak için PyYAML kütüphanesini içe aktarır
import rclpy # Temel ROS 2 Python istemci kütüphanesi
from rclpy.node import Node # ROS 2 düğüm (node) sınıfını içe aktarır
from rclpy.qos import ( # ROS 2 Kalite Servis (Quality of Service - QoS) ayarlarını içe aktarır
    QoSProfile, # Temel QoS profili yapısı
    QoSReliabilityPolicy, # Güvenilirlik politikasını ayarlar (örn: BEST_EFFORT, RELIABLE)
    QoSHistoryPolicy, # Tarihçe politikasını ayarlar (kaç mesajın tutulacağı)
    QoSDurabilityPolicy, # Dayanıklılık politikasını ayarlar (geç katılanların mesaj alıp almayacağı)
)
from rclpy.duration import Duration # ROS 2 zaman/süre tiplerini kullanmak için
from rclpy.parameter import Parameter # Dinamik parametreler için kullanılan yapı
from rcl_interfaces.msg import SetParametersResult # Parametre ayarlama işleminin sonucunu bildiren mesaj tipi

from ament_index_python.packages import get_package_share_directory # ROS paketlerinin paylaşılan dizinlerini bulmak için

import cv2 # OpenCV kütüphanesi (kamera yakalama ve görüntü işleme için)
from cv_bridge import CvBridge # OpenCV görüntülerini (numpy dizileri) ROS 2 Image mesajlarına çevirmek için
from sensor_msgs.msg import Image, CameraInfo # ROS 2 görüntü (Image) ve kamera kalibrasyon bilgisi (CameraInfo) mesaj tipleri
try: # camera_calibration_parsers kütüphanesini içe aktarmayı dener
    from camera_calibration_parsers import readCalibration # Kalibrasyon YAML dosyasını okuma fonksiyonu
    _HAS_CCP = True # Başarılı olursa bayrağı True yapar
except Exception: # Eğer kütüphane bulunamazsa (veya başka bir hata olursa)
    import yaml # PyYAML'ı içe aktar (manuel/fallback okuma için)
    _HAS_CCP = False # Bayrağı False yapar

# Ana kamera sürücü düğümü sınıfı tanımlanıyor, rclpy'nin Node sınıfından miras alır
class CameraDriverNode(Node):
    # Düğümün başlatıcı metodu
    def __init__(self) -> None:
        super().__init__("camera_driver") # ROS 2 düğümünü 'camera_driver' adıyla başlatır

        # ----------------------------
        # 1) Parametrelerin deklarasyonu
        # ----------------------------
        # Profil kaynakları (YAML dosyasını ve dosya içindeki hangi profili kullanacağımızı belirtiriz)
        self.declare_parameter("profiles_file", "")   # Parametre: Kamera profillerini içeren YAML dosyasının yolu
        self.declare_parameter("profile", "")         # Parametre: YAML dosyası içinden seçilecek profilin adı

        # Kamera konfigürasyonu (profil yoksa/override gerekirse kullanılacak parametreler)
        self.declare_parameter("camera_id", 0) # OpenCV'nin kullanacağı kamera/cihaz kimliği (genellikle tamsayı veya dosya yolu)
        self.declare_parameter("frame_rate", 30.0) # Görüntü yakalama hızı (Hz cinsinden)
        self.declare_parameter("camera_info_url", "file:///default/path/to/your/camera.yaml") # Kalibrasyon dosyasının URL'si
        self.declare_parameter("frame_id", "camera_optical_frame") # Yayınlanacak Image ve CameraInfo mesajlarının header'ındaki frame adı
        self.declare_parameter("image_width", 640) # Yakalanacak görüntünün genişliği (piksel)
        self.declare_parameter("image_height", 480) # Yakalanacak görüntünün yüksekliği (piksel)
        self.declare_parameter("fourcc", "MJPG")  # Kamera video codec'i (dört karakter kodu)

        # İç durum bayrakları (düğümün dahili durumunu takip etmek için)
        self.has_valid_calib_ = False # Geçerli bir kalibrasyon dosyasının yüklenip yüklenmediği
        self._calib_mismatch_warned_ = False  # Kalibrasyon boyutu ile akış boyutu uyuşmazlığı uyarısının verilip verilmediği
        self._first_info_published_ = False    # İlk görüntüyle birlikte CameraInfo'nun hemen yayınlanıp yayınlanmadığı

        # ----------------------------
        # 2) Parametreleri yükle + profil uygula (varsa)
        # ----------------------------
        self._load_parameters()       # ROS parametre sunucusundan deklarasyon/CLI ile gelen ilk değerleri çeker
        self._maybe_apply_profile()   # Eğer profiles_file ve profile parametreleri verilmişse, YAML profilini uygular
        self._load_parameters()       # Profil uygulandıktan sonraki son değerleri (özellikle URL normalizasyonu yapılmış halini) tekrar çeker

        # ----------------------------
        # 3) QoS profilleri
        # ----------------------------
        image_qos_profile = QoSProfile(depth=1) # Görüntü yayıncısı için temel QoS profilini oluşturur (buffer boyutu 1)
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT # En iyi çaba (BEST_EFFORT): Veri kaybı önemli değil, hız önemli
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST # Sadece son mesajı tut
        image_qos_profile.lifespan = Duration(seconds=0.1) # Yaşam süresi (eski frame'lerin hızlıca düşürülmesi için, düşük gecikme sağlar)

        cam_info_qos_profile = QoSProfile(depth=1) # CameraInfo yayıncısı için temel QoS profilini oluşturur (buffer boyutu 1)
        cam_info_qos_profile.reliability = QoSReliabilityPolicy.RELIABLE # Güvenilir: Mesajın alıcıya ulaştığı garanti edilmeli (kalibrasyon bilgisi için önemli)
        cam_info_qos_profile.history = QoSHistoryPolicy.KEEP_LAST # Sadece son mesajı tut
        cam_info_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL  # Geçici Yerel (TRANSIENT_LOCAL): Geç katılan düğümler, yayıncının tuttuğu son mesajı alır

        # ----------------------------
        # 4) Publisher'lar
        # ----------------------------
        # Görüntü (Image) yayıncısını oluşturur. Topic adı: "camera/image_raw"
        self.image_publisher_ = self.create_publisher(Image, "camera/image_raw", image_qos_profile)
        # Kamera Bilgisi (CameraInfo) yayıncısını oluşturur. Topic adı: "camera/camera_info"
        self.cam_info_publisher_ = self.create_publisher(CameraInfo, "camera/camera_info", cam_info_qos_profile)

        # ----------------------------
        # 5) Kalibrasyonu yükle
        # ----------------------------
        self.bridge_ = CvBridge() # CvBridge nesnesini oluşturur
        self.camera_info_msg_ = CameraInfo() # CameraInfo mesajı için bir placeholder oluşturur
        self._load_camera_info() # Kalibrasyon dosyasını yüklemeyi dener ve camera_info_msg_ değişkenini doldurur

        # ----------------------------
        # 6) Kamerayı başlat
        # ----------------------------
        self.camera_handle_ = self._initialize_camera() # OpenCV'yi kullanarak kamerayı başlatır
        if self.camera_handle_ is None: # Başlatma başarısız olursa
            raise RuntimeError("Kamera başlatılamadı.") # Kritik hata fırlatır ve düğümü sonlandırır

        # ----------------------------
        # 7) Zamanlayıcılar
        # ----------------------------
        self.image_timer_ = None # Görüntü yakalama zamanlayıcısı (Timer)
        self._create_image_timer()  # FPS'e göre ana görüntü yakalama döngüsü zamanlayıcısını kurar
        # CameraInfo yayınlayıcısını oluşturur (daha seyrek, her saniye bir kez)
        self.cam_info_timer_ = self.create_timer(1.0, self._cam_info_timer_cb) 

        # ----------------------------
        # 8) Dinamik parametre callback
        # ----------------------------
        # Çalışma anında parametre değişimi dinleyicisini ekler
        self.add_on_set_parameters_callback(self._parameters_cb)

        # Başlangıçta kamera durumunu loglar
        self.get_logger().info(
            f"Kamera hazır -> ID:{self.camera_id_} " # Kamera ID'si
            f"{self.image_width_}x{self.image_height_}@{self.frame_rate_} " # Çözünürlük ve FPS
            f"FOURCC:{self.fourcc_} frame_id:{self.frame_id_}" # FOURCC ve frame_id
        )

    # =========================================================
    # Profil uygulama + URL çözümleme
    # =========================================================
    # Mevcut profiles_file ve profile parametre değerlerini döner
    def _get_profiles_file_and_name(self) -> Tuple[str, str]:
        """profiles_file ve profile parametrelerini döndürür."""
        profiles_file = self.get_parameter("profiles_file").get_parameter_value().string_value
        profile_name  = self.get_parameter("profile").get_parameter_value().string_value
        return profiles_file, profile_name

    # camera_info_url değerini platformdan bağımsız mutlak yola dönüştürür (normalize eder)
    def _resolve_camera_info_url(self, url: str, profiles_file: str) -> str:
        """
        camera_info_url değerini platformdan bağımsız ve taşınabilir hale getirir.
        Dönen değer: 'file://<ABSOLUTE_PATH>'
        """
        if not url: # URL boşsa
            return "" # Boş döndür

        # 1) file:// doğrudan normalize et
        if url.startswith("file://"):
            abs_path = url.replace("file://", "", 1) # 'file://' kısmını kaldır
            # Çevre değişkenlerini, ~ işaretini ve göreli yolu mutlak yola çevirir
            abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(abs_path)))
            return f"file://{abs_path}" # 'file://' ekleyerek normalize edilmiş URL'yi döndürür

        # 2) package://<pkg>/... (ROS paket yolu çözümü)
        if url.startswith("package://"):
            try:
                pkg_and_rel = url.replace("package://", "", 1) # 'package://' kısmını kaldır
                pkg, rel = pkg_and_rel.split("/", 1) # Paket adı ve göreli yolu ayır
                pkg_share = get_package_share_directory(pkg) # Paketin paylaşılan dizinini bul
                abs_path = os.path.join(pkg_share, rel) # Mutlak yolu oluştur
                return f"file://{os.path.abspath(abs_path)}" # 'file://' ekleyerek mutlak URL'yi döndürür
            except Exception as e: # Paket bulunamazsa/başka bir hata olursa
                self.get_logger().error(f"package:// URL çözümlenemedi: {url} ({e})") # Hata logu basar
                return "" # Boş döndürür

        # 3) ${ENV} ve ~ genişlet
        expanded = os.path.expandvars(os.path.expanduser(url)) # Çevre değişkenlerini ve home dizini kısaltmasını genişletir

        # 4) Göreli yol ise, profiles_file konumuna göre çöz
        if not os.path.isabs(expanded) and profiles_file: # Eğer yol mutlak değilse ve bir profiles_file verilmişse
            base = Path(profiles_file).resolve().parent # profiles_file'ın bulunduğu dizini bulur
            expanded = str((base / expanded).resolve()) # Göreli yolu profiles_file'ın dizinine göre çözer

        return f"file://{os.path.abspath(expanded)}" # Son mutlak yolu 'file://' formatında döndürür

    # Eğer tanımlıysa, profil dosyasını okur ve parametreleri uygular
    def _maybe_apply_profile(self) -> None:
        """
        profiles_file + profile parametreleri verilmişse, ilgili profil sözlüğünü
        ROS parametrelerine uygular.
        """
        profiles_file, profile_name = self._get_profiles_file_and_name() # Profil dosya yolu ve adını alır
        if not profiles_file or not profile_name:
            return  # Profil kullanılmıyorsa fonksiyondan çıkar

        try:
            if not os.path.exists(profiles_file): # Profil dosyası mevcut değilse
                self.get_logger().error(f"profiles_file bulunamadı: {profiles_file}") # Hata logu basar
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

    # =========================================================
    # Parametre yükleme/doğrulama
    # =========================================================
    # ROS parametrelerini çeker, doğrular ve iç değişkenlere atar
    def _load_parameters(self) -> None:
        """ROS parametre sunucusundan değerleri çek, doğrula ve URL’yi normalize et."""
        # Parametre değerlerini çekme:
        self.camera_id_ = self.get_parameter("camera_id").get_parameter_value().integer_value
        self.frame_rate_ = float(self.get_parameter("frame_rate").get_parameter_value().double_value)
        self.camera_info_url_ = self.get_parameter("camera_info_url").get_parameter_value().string_value
        self.frame_id_ = self.get_parameter("frame_id").get_parameter_value().string_value
        self.image_width_ = int(self.get_parameter("image_width").get_parameter_value().integer_value)
        self.image_height_ = int(self.get_parameter("image_height").get_parameter_value().integer_value)
        self.fourcc_ = self.get_parameter("fourcc").get_parameter_value().string_value

        # FPS aralığı koruması (değer aralığı kontrolü)
        if not (1.0 <= self.frame_rate_ <= 120.0):
            self.get_logger().warning(
                f"frame_rate={self.frame_rate_} geçersiz. 30.0 olarak ayarlanıyor."
            )
            self.frame_rate_ = 30.0 # Geçersizse varsayılan değer atanır

        # Çözünürlük koruması (pozitif değer kontrolü)
        if self.image_width_ <= 0 or self.image_height_ <= 0:
            self.get_logger().warning(
                f"Geçersiz çözünürlük ({self.image_width_}x{self.image_height_}). 640x480 olarak ayarlanıyor."
            )
            self.image_width_, self.image_height_ = 640, 480 # Geçersizse varsayılan değerler atanır

        # FOURCC 4 karakter olmalı (format kontrolü)
        if len(self.fourcc_) < 4:
            self.get_logger().warning(f"FOURCC='{self.fourcc_}' geçersiz. 'MJPG' olarak ayarlanıyor.")
            self.fourcc_ = "MJPG"
        else:
            self.fourcc_ = self.fourcc_[:4] # Sadece ilk 4 karakteri alır (gerekiyorsa kısaltır)

        # camera_info_url’ü normalize et (profil/CLI'dan gelmiş olabilir)
        profiles_file, _ = self._get_profiles_file_and_name()
        if self.camera_info_url_: # URL boş değilse
            # URL çözümleme fonksiyonunu çağırarak mutlak 'file://' yoluna çevirir
            self.camera_info_url_ = self._resolve_camera_info_url(self.camera_info_url_, profiles_file)

    # =========================================================
    # Kamera başlatma
    # =========================================================
    # OpenCV'nin cap.set() sonucunu kontrol eden yardımcı metot
    def _set_prop(self, cap: cv2.VideoCapture, prop: int, value, label: str) -> None:
        """OpenCV cap.set sonucunu kontrol edip logla."""
        if not cap.set(prop, value):
            self.get_logger().warning(f"Kamera '{label}' ayarı başarısız/desteklenmiyor: {value}")

    # Kamerayı mevcut parametrelerle başlatan ana metot
    def _initialize_camera(self):
        """
        Kamerayı mevcut parametrelerle başlatır.
        Linux'ta V4L2 backend’i açık seçik isteyerek set(...) tutarlılığını artırır.
        """
        try:
            # OpenCV VideoCapture nesnesini oluşturur, arka uç (backend) olarak V4L2'yi kullanmayı dener
            cap = cv2.VideoCapture(self.camera_id_, cv2.CAP_V4L2)
            if not cap.isOpened(): # Kamera açılıp açılamadığını kontrol eder
                self.get_logger().error(f"Kamera açılamadı! ID: {self.camera_id_}")
                return None # Başarısız olursa None döndürür

            # Kamera özelliklerini ayarlar:
            self._set_prop(cap, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc_), "FOURCC") # FOURCC
            self._set_prop(cap, cv2.CAP_PROP_FRAME_WIDTH, self.image_width_, "WIDTH") # Genişlik
            self._set_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT, self.image_height_, "HEIGHT") # Yükseklik
            self._set_prop(cap, cv2.CAP_PROP_FPS, self.frame_rate_, "FPS") # FPS
            self._set_prop(cap, cv2.CAP_PROP_BUFFERSIZE, 1, "BUFFERSIZE") # Dahili buffer boyutunu 1'e ayarlar (düşük gecikme için)

            self.get_logger().info("Kamera başarıyla açıldı ve ayarlandı.") # Başarılı log
            return cap # Başarılı VideoCapture nesnesini döndürür
        except Exception as e: # Kritik bir hata oluşursa
            self.get_logger().fatal(f"Kamera başlatılırken kritik hata: {e}") # Kritik hata logu basar
            return None

    # =========================================================
    # Kalibrasyon yükleme
    # =========================================================
    # Kalibrasyon dosyasını (camera_info_url'den) okuyup CameraInfo mesajını doldurur
    def _load_camera_info(self):
        clean_url = self.camera_info_url_.replace("file://", "") # 'file://' önekini kaldırır
        try:
            if _HAS_CCP: # camera_calibration_parsers kütüphanesi varsa
                ok, camera_name, camera_info_msg = readCalibration(clean_url) # Kütüphane ile okumayı dener
                if ok: # Okuma başarılı olursa
                    self.camera_info_msg_ = camera_info_msg # Mesajı kaydeder
                    self.has_valid_calib_ = True # Geçerli kalibrasyon bayrağını True yapar
                    self.get_logger().info(f"Kalibrasyon yüklendi: {camera_name}") # Başarılı log
                    return # Fonksiyondan çıkar
                else: # Okuma başarısız olursa
                    self.get_logger().warning(f"CCP okuma başarısız: {clean_url}, PyYAML ile denenecek.")

            # Fallback: camera_calibration_parsers yoksa veya başarısız olursa PyYAML ile oku
            with open(clean_url, "r") as f: # Kalibrasyon dosyasını açar
                data = yaml.safe_load(f) or {} # YAML içeriğini yükler
            ci = CameraInfo() # Yeni bir CameraInfo mesajı oluşturur
            # YAML'daki ilgili alanları (image_width, image_height) doldurur
            ci.width  = int(data.get("image_width", 0) or 0)
            ci.height = int(data.get("image_height", 0) or 0)

            # Matris verilerini (K, D, R, P) çeker
            K = data.get("camera_matrix", {}).get("data")
            D = data.get("distortion_coefficients", {}).get("data")
            R = data.get("rectification_matrix", {}).get("data")
            P = data.get("projection_matrix", {}).get("data")
            # Matris verilerini CameraInfo mesajındaki listelere atar
            if K: ci.k = list(map(float, K))
            if D: ci.d = list(map(float, D))
            if R: ci.r = list(map(float, R))
            if P: ci.p = list(map(float, P))
            ci.distortion_model = data.get("distortion_model", "") # Distorsiyon modelini doldurur

            self.camera_info_msg_ = ci # Mesajı kaydeder
            self.has_valid_calib_ = True # Geçerli kalibrasyon bayrağını True yapar
            self.get_logger().info(f"Kalibrasyon (fallback) yüklendi: {clean_url}") # Başarılı log
        except Exception as e: # Yükleme sırasında hata olursa
            self.get_logger().error(f"Kalibrasyon yüklenirken hata: {e}") # Hata logu basar
            self.camera_info_msg_ = CameraInfo() # Boş bir CameraInfo mesajı atar
            self.has_valid_calib_ = False # Bayrağı False yapar

    # =========================================================
    # Zamanlayıcılar
    # =========================================================
    # Görüntü yakalama zamanlayıcısını oluşturur veya yeniden oluşturur
    def _create_image_timer(self) -> None:
        """FPS'e göre görüntü yakalama timer'ını oluştur/yeniden oluştur."""
        if self.image_timer_ is not None: # Zaten bir zamanlayıcı varsa
            self.destroy_timer(self.image_timer_) # Önce eskisini yok eder
        period = 1.0 / self.frame_rate_ # Periyot süresini FPS'e göre hesaplar (saniye cinsinden)
        self.image_timer_ = self.create_timer(period, self._image_timer_cb) # Yeni zamanlayıcıyı oluşturur
        self.get_logger().info(f"Görüntü zamanlayıcısı {self.frame_rate_:.1f} Hz.") # Loglar

    # Görüntü yakalama ve yayınlama geri çağırma fonksiyonu
    def _image_timer_cb(self) -> None:
        """Periyodik olarak görüntü alıp yayınla."""
        ret, frame = self.camera_handle_.read() # Kameradan bir frame okur
        if not ret: # Okuma başarısız olursa
            self.get_logger().warning("Kameradan görüntü alınamadı.") # Uyarı logu basar
            return # Fonksiyondan çıkar

        now = self.get_clock().now().to_msg() # Mevcut ROS zamanını alır
        msg = self.bridge_.cv2_to_imgmsg(frame, "bgr8") # OpenCV frame'ini ROS Image mesajına çevirir
        msg.header.stamp = now # Mesajın zaman damgasını ayarlar
        msg.header.frame_id = self.frame_id_ # Mesajın frame ID'sini ayarlar
        self.image_publisher_.publish(msg) # Image mesajını yayınlar

        # İlk görüntüyle birlikte CameraInfo’u bir defa hemen yayınlar (alıcılar için)
        if self.has_valid_calib_ and not self._first_info_published_:
            self.camera_info_msg_.header.stamp = now # CameraInfo mesajının zaman damgasını ayarlar
            self.camera_info_msg_.header.frame_id = self.frame_id_ # CameraInfo mesajının frame ID'sini ayarlar
            self.cam_info_publisher_.publish(self.camera_info_msg_) # CameraInfo mesajını yayınlar
            self._first_info_published_ = True # Bayrağı True yaparak bir daha yayınlanmasını engeller (bu döngüde)

    # CameraInfo yayınlama geri çağırma fonksiyonu
    def _cam_info_timer_cb(self) -> None:
        """CameraInfo’u periyodik olarak yayınla (TRANSIENT_LOCAL ile geç katılanlar da alır)."""
        if not self.has_valid_calib_: # Geçerli kalibrasyon yoksa
            return # Fonksiyondan çıkar

        # Kalibrasyon YAML çözünürlüğü ile canlı akış çözünürlüğü uyuşmazlığı kontrolü
        if (
            self.camera_info_msg_.width # Kalibrasyon genişliği tanımlıysa
            and self.camera_info_msg_.height # Kalibrasyon yüksekliği tanımlıysa
            and (self.camera_info_msg_.width != self.image_width_ # Genişlikler farklıysa
                 or self.camera_info_msg_.height != self.image_height_) # Veya yükseklikler farklıysa
        ):
            if not self._calib_mismatch_warned_: # Uyarı daha önce verilmediyse
                self.get_logger().warning( # Uyarıyı basar
                    f"CameraInfo ({self.camera_info_msg_.width}x{self.camera_info_msg_.height}) "
                    f"≠ capture ({self.image_width_}x{self.image_height_}). "
                    f"YAML’ı akış çözünürlüğüne göre güncelleyin."
                )
                self._calib_mismatch_warned_ = True # Bayrağı True yapar

        now = self.get_clock().now().to_msg() # Mevcut ROS zamanını alır
        self.camera_info_msg_.header.stamp = now # CameraInfo mesajının zaman damgasını ayarlar
        self.camera_info_msg_.header.frame_id = self.frame_id_ # CameraInfo mesajının frame ID'sini ayarlar
        self.cam_info_publisher_.publish(self.camera_info_msg_) # CameraInfo mesajını yayınlar

    # =========================================================
    # Dinamik parametre güncelleme
    # =========================================================
    # Parametre değişimleri için callback fonksiyonu
    def _parameters_cb(self, params: List[Parameter]) -> SetParametersResult:
        """
        Parametre değişimlerini güvenli uygula ve gerekli yeniden başlatmaları tetikle.
        """
        # Geçici olarak yeni değerleri tutmak için mevcut değerleri kullanır
        new_camera_id = self.camera_id_
        new_frame_rate = self.frame_rate_
        new_width = self.image_width_
        new_height = self.image_height_
        new_fourcc = self.fourcc_

        # Hangi yeniden başlatma işlemlerinin tetikleneceğini belirleyen bayraklar
        reinit_camera = False # Kamera yeniden başlatılacak mı? (ID, Çözünürlük, FOURCC değişirse)
        recreate_timer = False # Timer yeniden oluşturulacak mı? (FPS değişirse)
        rerun_profile = False # Profil yeniden uygulanacak mı? (profiles_file veya profile değişirse)

        for p in params: # Değiştirilen her bir parametre için döngü
            if p.name == "frame_rate": # FPS değişirse
                try:
                    v = float(p.value) # Değeri float'a çevir
                    if not (1.0 <= v <= 120.0): # Aralık kontrolü
                        # Başarısız olursa hata mesajı ile SetParametersResult döndürür
                        return SetParametersResult(successful=False, reason="frame_rate 1–120 aralığında olmalı")
                    new_frame_rate = v # Yeni değeri kaydeder
                    recreate_timer = True # Timer'ı yeniden oluşturma bayrağını True yapar
                except Exception: # Sayısal çevirme hatası olursa
                    return SetParametersResult(successful=False, reason="frame_rate sayısal olmalı")

            elif p.name in ("image_width", "image_height"): # Çözünürlük değişirse
                try:
                    iv = int(p.value) # Değeri tamsayıya çevir
                    if iv <= 0: # Pozitiflik kontrolü
                        return SetParametersResult(successful=False, reason=f"{p.name} > 0 olmalı")
                    if p.name == "image_width":
                        new_width = iv
                    else:
                        new_height = iv
                    reinit_camera = True # Kamerayı yeniden başlatma bayrağını True yapar
                except Exception:
                    return SetParametersResult(successful=False, reason=f"{p.name} tamsayı olmalı")

            elif p.name == "camera_id": # Kamera ID'si değişirse
                try:
                    new_camera_id = int(p.value) # Değeri tamsayıya çevir
                    reinit_camera = True # Kamerayı yeniden başlatma bayrağını True yapar
                except Exception:
                    return SetParametersResult(successful=False, reason="camera_id tamsayı olmalı")

            elif p.name == "fourcc": # FOURCC değişirse
                s = str(p.value) # Değeri string'e çevir
                if len(s) < 4: # Uzunluk kontrolü
                    return SetParametersResult(successful=False, reason="fourcc 4 karakter olmalı (örn. MJPG, YUYV)")
                new_fourcc = s[:4] # İlk 4 karakteri kaydeder
                reinit_camera = True # Kamerayı yeniden başlatma bayrağını True yapar

            elif p.name == "camera_info_url": # Kalibrasyon URL'si değişirse
                # Güncel profil dosyasını alır (URL çözümlemesi için)
                profiles_file, _ = self._get_profiles_file_and_name()
                # Yeni URL'yi normalize eder ve iç değişkene kaydeder
                self.camera_info_url_ = self._resolve_camera_info_url(str(p.value), profiles_file)
                self._load_camera_info() # Yeni URL ile kalibrasyonu tekrar yükler

            elif p.name == "frame_id": # Frame ID değişirse
                self.frame_id_ = str(p.value) # Yeni frame ID'yi kaydeder

            elif p.name in ("profiles_file", "profile"): # Profil ayarları değişirse
                rerun_profile = True # Profilin yeniden uygulanması bayrağını True yapar

        # Profil yeniden uygulanmak istenirse:
        if rerun_profile:
            self._maybe_apply_profile() # Profil uygulama metodunu çağırır (bu set_parameters'ı tetikleyebilir)
            # Profil sonrası kesinleşen son değerleri (ID, çözünürlük, FPS vb.) tekrar çeker
            self._load_parameters()
            reinit_camera = True # Profil değişimi genellikle kamera yeniden başlatmayı gerektirir
            recreate_timer = True # Profil değişimi genellikle FPS değişimini de içerir

        # Kamera yeniden başlatma gerekirse
        if reinit_camera:
            if self.camera_handle_ is not None: # Kamera açıksa
                self.camera_handle_.release() # Önce kamerayı serbest bırakır (kapatır)

            # Yeni değerleri iç değişkenlere atar
            self.camera_id_, self.image_width_, self.image_height_, self.fourcc_ = (
                new_camera_id,
                new_width,
                new_height,
                new_fourcc,
            )
            self.camera_handle_ = self._initialize_camera() # Kamerayı yeni ayarlarla başlatmayı dener
            if self.camera_handle_ is None: # Yeni ayarlarla başlatma başarısız olursa
                # Eski ayarlara geri dönmeyi dener (best-effort)
                self._load_parameters()
                self.camera_handle_ = self._initialize_camera()
                # Başarısız sonucu döndürür
                return SetParametersResult(successful=False, reason="Kamera yeni ayarlarla açılamadı")

        # FPS değişimi timer’ı etkiler
        if recreate_timer:
            self.frame_rate_ = new_frame_rate # Yeni FPS değerini kaydeder
            self._create_image_timer() # Timer'ı yeniden oluşturur

        return SetParametersResult(successful=True) # Tüm parametreler başarıyla ayarlandı

    # =========================================================
    # Temizlik
    # =========================================================
    # Düğüm kapanırken çağrılan temizlik metodu
    def destroy_node(self) -> None:
        """Düğüm kapanırken kaynakları serbest bırak."""
        self.get_logger().info("Kamera kapatılıyor.")
        # Kamera handle'ı varsa (açıksa)
        if hasattr(self, "camera_handle_") and self.camera_handle_ is not None:
            self.camera_handle_.release() # OpenCV kamera kaynağını serbest bırakır
        super().destroy_node() # Üst sınıfın destroy_node metodunu çağırır



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
            node.destroy_node() # Düğümün temizlik metodunu çağırır
        if rclpy.ok(): # rclpy hala çalışıyorsa (henüz kapanmamışsa)
            rclpy.shutdown() # ROS 2 Python istemcisini kapatır

if __name__ == "__main__":
    main()
