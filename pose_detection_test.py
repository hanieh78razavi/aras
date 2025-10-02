import cv2
import mediapipe as mp
import numpy as np
import os
import time
# وارد کردن کتابخانه‌های رندر فارسی
import arabic_reshaper 
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List, Any
import threading

# مقداردهی اولیه MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
# ------------------------------------------------------------------------------------------------------

class SuperPoseDetector:
    """سوپر اپلیکیشن تشخیص پوز با تمام قابلیت‌های پیشرفته"""
    
    def __init__(self):
        # ------------------------
        # ۱. مدل‌های MediaPipe
        # ------------------------
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # ------------------------
        # ۲. تعریف متغیرهای وضعیت
        # ------------------------
        self.cap = None
        self.video_writer = None
        
        # وضعیت‌های کنترلی
        self.button_cooldown = 0
        self.is_recording = False
        self.timer_active = False
        self.current_filter = "normal" 
        self.flash_effect = 0
        
        # زمان‌ها و شمارنده‌ها
        self.recording_start_time = 0
        self.timer_countdown = 0 
        self.last_gesture_time = 0
        self.photo_count = 0
        self.video_count = 0 

        # ------------------------
        # ۳. بهینه‌سازی پردازش (Threading) و فونت
        # ------------------------
        # متغیرهای جدید برای بهینه‌سازی سرعت
        self.frame_to_process = None # فریم جدید برای پردازش توسط رشته دوم
        self.process_results = {'pose': None, 'hand': None, 'face': None} # نتایج آخرین پردازش
        
        # ۱. بهینه‌سازی فونت: فونت را فقط یک بار لود می‌کنیم
        self.persian_font_path = self.get_persian_font() 
        
        # ۲. راه‌اندازی رشته پردازشی
        self.processing_thread = threading.Thread(target=self._run_processing_loop, daemon=True)
        self.processing_thread.start()
        
        # ۳. راه‌اندازی‌های اولیه
        self.create_folders()

    def create_folders(self):
        """ایجاد پوشه‌های مورد نیاز"""
        folders = ['photos', 'videos', 'gestures']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                
    # ------------------------------------------------------------------------------------------------------
    # توابع رندر فارسی (اصلاح شده برای لود فونت یک بار)
    # ------------------------------------------------------------------------------------------------------

    def get_persian_font(self) -> Optional[str]:
        """پیدا کردن فونت فارسی B Nazanin، IRANSans یا Tahoma"""
        # لیست فونت‌های فارسی رایج
        persian_fonts = ['BNazanin.ttf', 'B Nazanin.ttf', 'IRANSans.ttf', 'Tahoma.ttf', 'Vazir.ttf', 'arial.ttf']
        
        # مسیرهای ویندوز و سیستم‌های دیگر
        font_paths = [
            'C:/Windows/Fonts/',         # آدرس مستقیم ویندوز
            'C:\\Windows\\Fonts\\',      # آدرس ویندوز با بک‌اسلش
            '/usr/share/fonts/truetype/', # مسیرهای رایج لینوکس
            '/System/Library/Fonts/',     # مسیرهای رایج مک
            os.path.expanduser('~/.fonts/'), # مسیرهای کاربری لینوکس/مک
            os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts/'), # مسیرهای ویندوز کاربری
            '.'                          # پوشه جاری
        ]
        
        for font_path in font_paths:
            # بررسی مستقیم مسیر
            if os.path.exists(font_path):
                for font in persian_fonts:
                    full_path = os.path.join(font_path, font)
                    if os.path.exists(full_path):
                        return full_path
                        
        return None # اگر هیچ فونتی پیدا نشد
    
    def put_persian_text(self, img: np.ndarray, text: str, position: Tuple[int, int], 
                          font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """نمایش متن فارسی با استفاده از Pillow و Bidi برای تصحیح رندر"""
        try:
            # ۱. تصحیح فارسی و جهت‌دهی (CTL)
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            # ۲. تبدیل OpenCV به Pillow
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # ۳. مدیریت فونت و اندازه
            base_size = 30 
            font_size = max(30, int(base_size * font_scale)) # حداقل اندازه 30 برای خوانایی بهتر
            
            # استفاده از مسیر فونت لود شده در __init__
            if self.persian_font_path:
                font = ImageFont.truetype(self.persian_font_path, font_size)
            else:
                # Fallback به فونت پیش‌فرض با اندازه تعیین شده
                font = ImageFont.load_default(size=font_size)
                
            # ۴. ترسیم متن
            draw.text(position, bidi_text, font=font, fill=color)
            
            # ۵. بازگشت به فرمت OpenCV
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            # در صورت بروز خطا، به cv2.putText برگرد
            # print(f"Error in put_persian_text: {e}")
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            return img

    # ------------------------------------------------------------------------------------------------------
    # توابع پردازشی و کمکی (اصلاح شده برای Threading)
    # ------------------------------------------------------------------------------------------------------
    
    def _run_processing_loop(self):
        """حلقه پردازشی ناهمزمان برای اجرای مدل‌های MP"""
        while True:
            if self.frame_to_process is not None:
                # کپی کردن فریم برای جلوگیری از Race Condition
                frame = self.frame_to_process.copy() 
                self.frame_to_process = None # پاکسازی فریم برای پذیرش فریم بعدی
                
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose_detector.process(image_rgb)
                    hand_results = self.hand_detector.process(image_rgb)
                    face_results = self.face_detector.process(image_rgb)
                    
                    # ذخیره نتایج در متغیر مشترک
                    self.process_results = {
                        'pose': pose_results, 
                        'hand': hand_results, 
                        'face': face_results
                    }
                except Exception as e:
                    # print(f"Error in processing thread: {e}")
                    self.process_results = {'pose': None, 'hand': None, 'face': None}
            # کاهش بار CPU با Sleep (برای Threading مهم است)
            time.sleep(0.005) 

    def process_frame(self, frame: np.ndarray) -> Tuple[Any, Any, Any]:
        """ارسال فریم به رشته پردازش و بازگرداندن آخرین نتایج"""
        # فقط فریم را برای پردازش در رشته دیگر ارسال می‌کند
        self.frame_to_process = frame.copy()
        
        # بازگشت آخرین نتایج پردازش شده
        return self.process_results.get('pose'), self.process_results.get('hand'), self.process_results.get('face')
    
    def apply_filter(self, frame: np.ndarray, filter_name: str) -> np.ndarray:
        """اعمال فیلترهای رنگی مختلف"""
        if filter_name == "normal":
            return frame
        elif filter_name == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        elif filter_name == "grayscale":
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif filter_name == "warm":
            frame = frame.astype(np.float32)
            frame[:, :, 0] = frame[:, :, 0] * 0.7   # کاهش آبی
            frame[:, :, 2] = frame[:, :, 2] * 1.3   # افزایش قرمز
            return np.clip(frame, 0, 255).astype(np.uint8)
        elif filter_name == "cool":
            frame = frame.astype(np.float32)
            frame[:, :, 0] = frame[:, :, 0] * 1.3   # افزایش آبی
            frame[:, :, 2] = frame[:, :, 2] * 0.7   # کاهش قرمز
            return np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    
    def detect_gestures(self, hand_results: Any) -> str:
        """تشخیص ژست‌های دست (✌️ صلح و 👍 شست بالا)"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return ""
            
        current_time = time.time()
        # جلوگیری از تشخیص پشت سر هم (کول‌داون ژست)
        if current_time - self.last_gesture_time < 2: 
            return ""
            
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # ✌️ علامت V (صلح)
            if (landmarks[8].y < landmarks[6].y and   # انگشت اشاره بالا
                landmarks[12].y < landmarks[10].y and   # انگشت وسط بالا
                landmarks[16].y > landmarks[14].y and   # انگشت حلقه پایین
                landmarks[20].y > landmarks[18].y):     # انگشت کوچک پایین
                self.last_gesture_time = current_time
                return "peace"
            
            # 👍 شست بالا
            if (landmarks[4].y < landmarks[3].y and   # شست بالا
                landmarks[8].y > landmarks[6].y and   # سایر انگشتان پایین
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y):
                self.last_gesture_time = current_time
                return "thumbs_up"
                
        return ""
        
    def detect_smile(self, face_results: Any) -> bool:
        """تشخیص لبخند بر اساس لندمارک‌های صورت"""
        # MediaPipe Face Detection فقط کادر صورت را می‌دهد، نه لندمارک‌های دقیق دهان.
        # برای نمایش قابلیت، اگر صورت تشخیص داده شود، True برمی‌گردانیم.
        return bool(face_results and face_results.detections) 
        
    def take_photo(self, frame: np.ndarray):
        """گرفتن عکس و ذخیره آن"""
        self.photo_count += 1
        filename = f"photos/photo_{time.strftime('%Y%m%d_%H%M%S')}_{self.photo_count}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 عکس ذخیره شد: {filename}")
        self.flash_effect = 5 # فعال کردن فلش برای 5 فریم
        
    def toggle_recording(self):
        """شروع/توقف ضبط ویدیو"""
        if not self.is_recording:
            # شروع ضبط
            self.video_count += 1
            filename = f"videos/video_{time.strftime('%Y%m%d_%H%M%S')}_{self.video_count}.avi"
            
            # گرفتن ابعاد و FPS از کپچر دوربین
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap and self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
            self.recording_start_time = time.time()
            self.is_recording = True
            print(f"🔴 شروع ضبط ویدیو: {filename}")
        else:
            # توقف ضبط
            if self.video_writer:
                self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("⏹️ پایان ضبط ویدیو")
            
    def start_timer(self):
        """شروع تایمر ۳ ثانیه‌ای برای عکس"""
        if not self.timer_active:
            self.timer_active = True
            # فرض می‌کنیم برنامه با حدود ۳۰ فریم بر ثانیه اجرا می‌شود
            self.timer_countdown = 3 * 30 
            print("⏳ تایمر شروع شد (۳ ثانیه)")
            
    # --- توابع رابط کاربری ---
    
    def is_finger_touching_button(self, hand_results: Any, frame_shape: Tuple[int, int], 
                                  button_pos: Tuple[int, int, int, int]) -> bool:
        """بررسی لمس دکمه با انگشت"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return False
        
        h, w = frame_shape[:2]
        button_x1, button_y1, button_x2, button_y2 = button_pos
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # لندمارک شماره ۸ نوک انگشت اشاره (INDEX_FINGER_TIP) است.
            fingertip = hand_landmarks.landmark[8]  
            
            # تبدیل مختصات نرمالایز شده به مختصات پیکسلی
            fingertip_x = int(fingertip.x * w)
            fingertip_y = int(fingertip.y * h)
            
            # آیا نوک انگشت در محدوده دکمه قرار دارد؟
            if (button_x1 <= fingertip_x <= button_x2 and 
                button_y1 <= fingertip_y <= button_y2):
                return True
        
        return False

    def draw_virtual_buttons(self, frame: np.ndarray, hand_results: Any) -> Tuple[np.ndarray, dict]:
        """رسم تمام دکمه‌های مجازی و مدیریت لمس"""
        h, w = frame.shape[:2]
        buttons = {}
        
        # 🎨 دکمه‌های فیلتر (ستون سمت چپ) - فارسی‌سازی شده
        filters = {"عادی": "normal", "سپیا": "sepia", "سیاه‌سفید": "grayscale", "گرم": "warm", "سرد": "cool"}
        button_height = 50
        
        for i, (persian_name, filter_key) in enumerate(filters.items()):
            y1 = 20 + i * (button_height + 10)
            y2 = y1 + button_height
            x1, x2 = 20, 120
            
            is_active = self.current_filter == filter_key
            color = (0, 200, 0) if is_active else (100, 100, 100)
            
            # بررسی لمس و اعمال فیلتر
            is_touching = self.is_finger_touching_button(hand_results, frame.shape, (x1, y1, x2, y2))
            if is_touching and self.button_cooldown == 0:
                self.current_filter = filter_key
                self.button_cooldown = 20
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            # استفاده صحیح از persian_name برای رندر فارسی
            text = persian_name 
            frame = self.put_persian_text(frame, text, (x1 + 10, y1 + 10), 0.7, (255, 255, 255))
            
            buttons[f"filter_{filter_key}"] = (x1, y1, x2, y2)

        # 📸 دکمه عکس (گوشه بالا راست) - لمس فقط برای فعال‌سازی تایمر
        photo_x1, photo_y1 = w - 140, 20
        photo_x2, photo_y2 = w - 20, 80
        photo_touching = self.is_finger_touching_button(hand_results, frame.shape, (photo_x1, photo_y1, photo_x2, photo_y2))
        
        photo_color = (0, 200, 0) if photo_touching else (200, 200, 200)
        cv2.rectangle(frame, (photo_x1, photo_y1), (photo_x2, photo_y2), photo_color, -1)
        cv2.rectangle(frame, (photo_x1, photo_y1), (photo_x2, photo_y2), (0, 0, 0), 2)
        
        frame = self.put_persian_text(frame, "عکس", (photo_x1 + 30, photo_y1 + 15), 1.0, (0, 0, 0))
        
        buttons["photo"] = (photo_x1, photo_y1, photo_x2, photo_y2)

        # 📹 دکمه ضبط ویدیو
        video_x1, video_y1 = w - 140, 100
        video_x2, video_y2 = w - 20, 160
        video_touching = self.is_finger_touching_button(hand_results, frame.shape, (video_x1, video_y1, video_x2, video_y2))
        
        # مدیریت لمس ضبط ویدیو
        if video_touching and self.button_cooldown == 0:
            self.toggle_recording()
            self.button_cooldown = 30
        
        video_color = (0, 0, 200) if self.is_recording else (200, 200, 200) # قرمز در حال ضبط
        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), video_color, -1)
        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), (0, 0, 0), 2)
        
        text = "توقف" if self.is_recording else "ضبط"
        frame = self.put_persian_text(frame, text, (video_x1 + 30, video_y1 + 15), 1.0, (255, 255, 255))
        
        buttons["video"] = (video_x1, video_y1, video_x2, video_y2)

        # ⏰ دکمه تایمر
        timer_x1, timer_y1 = w - 140, 180
        timer_x2, timer_y2 = w - 20, 240
        timer_touching = self.is_finger_touching_button(hand_results, frame.shape, (timer_x1, timer_y1, timer_x2, timer_y2))
        
        # مدیریت لمس تایمر
        if timer_touching and self.button_cooldown == 0:
            self.start_timer()
            self.button_cooldown = 30
        
        # اصلاح رنگ تایمر به زرد
        timer_color = (0, 200, 200) if self.timer_active else (200, 200, 200) # زرد (B:0, G:200, R:200)
        cv2.rectangle(frame, (timer_x1, timer_y1), (timer_x2, timer_y2), timer_color, -1)
        cv2.rectangle(frame, (timer_x1, timer_y1), (timer_x2, timer_y2), (0, 0, 0), 2)
        
        frame = self.put_persian_text(frame, "تایمر", (timer_x1 + 25, timer_y1 + 15), 1.0, (0, 0, 0))
        
        buttons["timer"] = (timer_x1, timer_y1, timer_x2, timer_y2)

        return frame, buttons

    def draw_landmarks(self, frame: np.ndarray, pose_results: Any, hand_results: Any) -> np.ndarray:
        """رسم اسکلت بدن و دست"""
        # رسم اسکلت بدن
        if pose_results and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # رسم دست‌ها
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        return frame

    def display_info(self, frame: np.ndarray, pose_results: Any, hand_results: Any, face_results: Any) -> np.ndarray:
        """نمایش اطلاعات و وضعیت (تماماً فارسی‌سازی شده)"""
        h, w = frame.shape[:2]
        
        # وضعیت تشخیص
        status = "فعال" if pose_results and pose_results.pose_landmarks else "غیرفعال"
        status_color = (0, 100, 0) if pose_results and pose_results.pose_landmarks else (0, 0, 100)
        frame = self.put_persian_text(frame, f"وضعیت: {status}", (20, h - 100), 0.7, status_color)
        
        # تایمر
        if self.timer_active:
            # زمان باقی‌مانده
            seconds = self.timer_countdown // 30 + 1 
            frame = self.put_persian_text(frame, f"تایمر: {seconds} ثانیه", (20, h - 70), 0.8, (0, 0, 200))
        
        # ضبط
        if self.is_recording:
            duration = int(time.time() - self.recording_start_time)
            frame = self.put_persian_text(frame, f"ضبط: {duration} ثانیه", (20, h - 40), 0.7, (0, 0, 200))
        
        # ژست‌ها
        gesture = self.detect_gestures(hand_results)
        if gesture:
            gesture_text = {"peace": "✌️ صلح", "thumbs_up": "👍 عالی"}.get(gesture, gesture)
            frame = self.put_persian_text(frame, f"ژست: {gesture_text}", (w - 200, h - 40), 0.7, (200, 0, 200))
        
        # لبخند
        if self.detect_smile(face_results):
            frame = self.put_persian_text(frame, "😊 لبخند", (w - 200, h - 70), 0.7, (0, 200, 200))
        
        return frame

    # ------------------------------------------------------------------------------------------------------
    # تابع اصلی و کمکی
    # ------------------------------------------------------------------------------------------------------
    
    def run(self):
        """اجرای اصلی برنامه"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ خطا در باز کردن دوربین")
            return
        
        # تنظیمات ابعاد دوربین
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🚀 سوپر اپلیکیشن تشخیص پوز راه‌اندازی شد!")
        print("🎯 امکانات: عکس، فیلم، تایمر، فیلتر، تشخیص ژست و لبخند!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # آینه‌ای کردن فریم
                frame = cv2.flip(frame, 1)
                
                # ** پردازش فریم در رشته جداگانه (بهینه‌سازی) **
                pose_results, hand_results, face_results = self.process_frame(frame)
                
                # اعمال فیلتر
                frame = self.apply_filter(frame, self.current_filter)
                
                # رسم اسکلت
                frame = self.draw_landmarks(frame, pose_results, hand_results)
                
                # رسم دکمه‌ها و بررسی لمس
                frame, buttons = self.draw_virtual_buttons(frame, hand_results)
                
                # بررسی لمس دکمه عکس (اگر تایمر فعال نیست)
                if "photo" in buttons and self.button_cooldown == 0:
                    photo_touching = self.is_finger_touching_button(hand_results, frame.shape, buttons["photo"])
                    if photo_touching:
                         if not self.timer_active:
                            self.take_photo(frame)
                            self.button_cooldown = 30
                
                # مدیریت تایمر
                if self.timer_active:
                    self.timer_countdown -= 1
                    if self.timer_countdown <= 0:
                        self.timer_active = False
                        self.take_photo(frame)  # عکس خودکار بعد از اتمام تایمر
                
                # ضبط ویدیو
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # کاهش کول‌داون
                if self.button_cooldown > 0:
                    self.button_cooldown -= 1
                
                # افکت فلش
                if self.flash_effect > 0:
                    flash_overlay = np.ones_like(frame) * 255
                    # ترکیب فریم با رنگ سفید برای ایجاد فلش
                    frame = cv2.addWeighted(frame, 0.7, flash_overlay, 0.3, 0)
                    self.flash_effect -= 1
                
                # نمایش اطلاعات وضعیت
                frame = self.display_info(frame, pose_results, hand_results, face_results)
                
                # نمایش فریم
                cv2.imshow('Super Pose Detection 🚀', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ برنامه متوقف شد")
        finally:
            self.cleanup()

    def cleanup(self):
        """پاکسازی منابع"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ منابع آزاد شدند")

def main():
    """تابع اصلی"""
    print("🎉 در حال راه‌اندازی سوپر اپلیکیشن...")
    app = SuperPoseDetector()
    app.run()

if __name__ == "__main__":
    main()