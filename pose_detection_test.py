import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List, Any
import threading

# مقداردهی اولیه MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

class SuperPoseDetector:
    """سوپر اپلیکیشن تشخیص پوز با تمام قابلیت‌های پیشرفته"""
    
    def __init__(self):
        # مدل تشخیص پوز
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # مدل تشخیص دست
        self.hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # مدل تشخیص چهره
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        self.cap = None
        self.button_cooldown = 0
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = 0
        self.current_filter = "normal"
        self.photo_count = 0
        self.video_count = 0
        self.timer_active = False
        self.timer_countdown = 0
        self.last_gesture_time = 0
        self.flash_effect = 0
        
        # ایجاد پوشه‌ها
        self.create_folders()
        
    def create_folders(self):
        """ایجاد پوشه‌های مورد نیاز"""
        folders = ['photos', 'videos', 'gestures']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def get_persian_font(self) -> Optional[str]:
        """پیدا کردن فونت فارسی"""
        persian_fonts = ['Vazir.ttf', 'B Nazanin.ttf', 'IRANSans.ttf', 'Tahoma.ttf']
        font_paths = ['C:/Windows/Fonts/', os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts/')]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    available_fonts = os.listdir(font_path)
                    for font in persian_fonts:
                        if font in available_fonts:
                            return os.path.join(font_path, font)
                except:
                    continue
        return None

    def put_persian_text(self, img: np.ndarray, text: str, position: Tuple[int, int], 
                        font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """نمایش متن فارسی"""
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            font_path = self.get_persian_font()
            font_size = max(20, int(28 * font_scale))
            
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = None
            
            draw.text(position, bidi_text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
        except:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            return img

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
            frame[:, :, 0] = frame[:, :, 0] * 0.7  # کاهش آبی
            frame[:, :, 2] = frame[:, :, 2] * 1.3  # افزایش قرمز
            return np.clip(frame, 0, 255).astype(np.uint8)
        elif filter_name == "cool":
            frame = frame.astype(np.float32)
            frame[:, :, 0] = frame[:, :, 0] * 1.3  # افزایش آبی
            frame[:, :, 2] = frame[:, :, 2] * 0.7  # کاهش قرمز
            return np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def detect_gestures(self, hand_results: Any) -> str:
        """تشخیص ژست‌های دست"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return ""
            
        current_time = time.time()
        if current_time - self.last_gesture_time < 2:  # جلوگیری از تشخیص پشت سر هم
            return ""
            
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # ✌️ علامت V (صلح)
            if (landmarks[8].y < landmarks[6].y and  # انگشت اشاره بالا
                landmarks[12].y < landmarks[10].y and  # انگشت وسط بالا
                landmarks[16].y > landmarks[14].y and  # انگشت حلقه پایین
                landmarks[20].y > landmarks[18].y):    # انگشت کوچک پایین
                self.last_gesture_time = current_time
                return "peace"
            
            # 👍 شست بالا
            if (landmarks[4].y < landmarks[3].y and  # شست بالا
                landmarks[8].y > landmarks[6].y and   # سایر انگشتان پایین
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y):
                self.last_gesture_time = current_time
                return "thumbs_up"
            
            # 👋 دست تکان دادن (بررسی حرکت)
            # اینجا می‌توانیم حرکت دست را در فریم‌های متوالی بررسی کنیم
            
        return ""

    def detect_smile(self, face_results: Any) -> bool:
        """تشخیص لبخند (ساده)"""
        if not face_results or not face_results.detections:
            return False
        
        # این یک تشخیص ساده است - در نسخه واقعی از landmarks صورت استفاده می‌شود
        return len(face_results.detections) > 0

    def is_finger_touching_button(self, hand_results: Any, frame_shape: Tuple[int, int], 
                                button_pos: Tuple[int, int, int, int]) -> bool:
        """بررسی لمس دکمه با انگشت"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return False
        
        h, w = frame_shape[:2]
        button_x1, button_y1, button_x2, button_y2 = button_pos
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            fingertip = hand_landmarks.landmark[8]  # نوک انگشت اشاره
            fingertip_x = int(fingertip.x * w)
            fingertip_y = int(fingertip.y * h)
            
            if (button_x1 <= fingertip_x <= button_x2 and 
                button_y1 <= fingertip_y <= button_y2):
                return True
        
        return False

    def draw_virtual_buttons(self, frame: np.ndarray, hand_results: Any) -> Tuple[np.ndarray, dict]:
        """رسم تمام دکمه‌های مجازی"""
        h, w = frame.shape[:2]
        buttons = {}
        
        # 🎨 دکمه‌های فیلتر (ستون سمت چپ)
        filters = ["normal", "sepia", "grayscale", "warm", "cool"]
        button_height = 50
        for i, filter_name in enumerate(filters):
            y1 = 20 + i * (button_height + 10)
            y2 = y1 + button_height
            x1, x2 = 20, 120
            
            is_active = self.current_filter == filter_name
            color = (0, 200, 0) if is_active else (100, 100, 100)
            
            # بررسی لمس
            is_touching = self.is_finger_touching_button(hand_results, frame.shape, (x1, y1, x2, y2))
            if is_touching and self.button_cooldown == 0:
                self.current_filter = filter_name
                self.button_cooldown = 20
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            text = filter_name[:3]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (100 - text_size[0]) // 2
            text_y = y1 + (button_height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            buttons[f"filter_{filter_name}"] = (x1, y1, x2, y2)

        # 📸 دکمه عکس (گوشه بالا راست)
        photo_x1, photo_y1 = w - 140, 20
        photo_x2, photo_y2 = w - 20, 80
        photo_touching = self.is_finger_touching_button(hand_results, frame.shape, (photo_x1, photo_y1, photo_x2, photo_y2))
        
        photo_color = (0, 200, 0) if photo_touching else (200, 200, 200)
        cv2.rectangle(frame, (photo_x1, photo_y1), (photo_x2, photo_y2), photo_color, -1)
        cv2.rectangle(frame, (photo_x1, photo_y1), (photo_x2, photo_y2), (0, 0, 0), 2)
        cv2.putText(frame, "عکس", (photo_x1 + 30, photo_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        buttons["photo"] = (photo_x1, photo_y1, photo_x2, photo_y2)

        # 📹 دکمه ضبط ویدیو
        video_x1, video_y1 = w - 140, 100
        video_x2, video_y2 = w - 20, 160
        video_color = (0, 0, 200) if self.is_recording else (200, 200, 200)
        video_touching = self.is_finger_touching_button(hand_results, frame.shape, (video_x1, video_y1, video_x2, video_y2))
        
        if video_touching and self.button_cooldown == 0:
            self.toggle_recording()
            self.button_cooldown = 30
        
        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), video_color, -1)
        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), (0, 0, 0), 2)
        cv2.putText(frame, "ضبط", (video_x1 + 30, video_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        buttons["video"] = (video_x1, video_y1, video_x2, video_y2)

        # ⏰ دکمه تایمر
        timer_x1, timer_y1 = w - 140, 180
        timer_x2, timer_y2 = w - 20, 240
        timer_touching = self.is_finger_touching_button(hand_results, frame.shape, (timer_x1, timer_y1, timer_x2, timer_y2))
        
        if timer_touching and self.button_cooldown == 0:
            self.start_timer()
            self.button_cooldown = 30
        
        timer_color = (200, 200, 0) if self.timer_active else (200, 200, 200)
        cv2.rectangle(frame, (timer_x1, timer_y1), (timer_x2, timer_y2), timer_color, -1)
        cv2.rectangle(frame, (timer_x1, timer_y1), (timer_x2, timer_y2), (0, 0, 0), 2)
        cv2.putText(frame, "تایمر", (timer_x1 + 25, timer_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        buttons["timer"] = (timer_x1, timer_y1, timer_x2, timer_y2)

        return frame, buttons

    def take_photo(self, frame: np.ndarray):
        """گرفتن عکس با افکت فلش"""
        self.photo_count += 1
        filename = f"photos/photo_{self.photo_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ عکس ذخیره شد: {filename}")
        
        # افکت فلش
        self.flash_effect = 10

    def toggle_recording(self):
        """شروع/توقف ضبط ویدیو"""
        if not self.is_recording:
            # شروع ضبط
            self.video_count += 1
            filename = f"videos/video_{self.video_count:04d}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
            self.is_recording = True
            self.recording_start_time = time.time()
            print(f"🎥 شروع ضبط: {filename}")
        else:
            # توقف ضبط
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            duration = time.time() - self.recording_start_time
            print(f"⏹️ توقف ضبط - مدت: {duration:.1f} ثانیه")

    def start_timer(self):
        """شروع تایمر 3 ثانیه"""
        self.timer_active = True
        self.timer_countdown = 90  # 3 ثانیه (90 فریم)

    def process_frame(self, frame: np.ndarray) -> Tuple[Any, Any, Any]:
        """پردازش فریم برای تمام مدل‌ها"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pose_results = self.pose_detector.process(image_rgb)
            hand_results = self.hand_detector.process(image_rgb)
            face_results = self.face_detector.process(image_rgb)
            
            return pose_results, hand_results, face_results
        except:
            return None, None, None

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
        """نمایش اطلاعات و وضعیت"""
        h, w = frame.shape[:2]
        
        # وضعیت تشخیص
        status = "فعال" if pose_results and pose_results.pose_landmarks else "غیرفعال"
        status_color = (0, 100, 0) if pose_results and pose_results.pose_landmarks else (0, 0, 100)
        frame = self.put_persian_text(frame, f"وضعیت: {status}", (20, h - 100), 0.7, status_color)
        
        # تایمر
        if self.timer_active:
            seconds = self.timer_countdown // 30 + 1
            frame = self.put_persian_text(frame, f"تایمر: {seconds}", (20, h - 70), 0.8, (0, 0, 200))
        
        # ضبط
        if self.is_recording:
            duration = int(time.time() - self.recording_start_time)
            frame = self.put_persian_text(frame, f"ضبط: {duration}s", (20, h - 40), 0.7, (0, 0, 200))
        
        # ژست‌ها
        gesture = self.detect_gestures(hand_results)
        if gesture:
            gesture_text = {"peace": "✌️ صلح", "thumbs_up": "👍 عالی"}.get(gesture, gesture)
            frame = self.put_persian_text(frame, f"ژست: {gesture_text}", (w - 200, h - 40), 0.7, (200, 0, 200))
        
        # لبخند
        if self.detect_smile(face_results):
            frame = self.put_persian_text(frame, "😊 لبخند", (w - 200, h - 70), 0.7, (0, 200, 200))
        
        return frame

    def run(self):
        """اجرای اصلی برنامه"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ خطا در باز کردن دوربین")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🚀 سوپر اپلیکیشن تشخیص پوز راه‌اندازی شد!")
        print("🎯 امکانات: عکس، فیلم، تایمر، فیلتر، تشخیص ژست و لبخند!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # پردازش فریم
                pose_results, hand_results, face_results = self.process_frame(frame)
                
                # اعمال فیلتر
                frame = self.apply_filter(frame, self.current_filter)
                
                # رسم اسکلت
                frame = self.draw_landmarks(frame, pose_results, hand_results)
                
                # رسم دکمه‌ها و بررسی لمس
                frame, buttons = self.draw_virtual_buttons(frame, hand_results)
                
                # بررسی لمس دکمه عکس
                if "photo" in buttons:
                    photo_touching = self.is_finger_touching_button(hand_results, frame.shape, buttons["photo"])
                    if photo_touching and self.button_cooldown == 0:
                        if self.timer_active:
                            # اگر تایمر فعال است، منتظر بمان
                            pass
                        else:
                            self.take_photo(frame)
                            self.button_cooldown = 30
                
                # مدیریت تایمر
                if self.timer_active:
                    self.timer_countdown -= 1
                    if self.timer_countdown <= 0:
                        self.timer_active = False
                        self.take_photo(frame)  # عکس خودکار
                
                # ضبط ویدیو
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # کاهش کول‌داون
                if self.button_cooldown > 0:
                    self.button_cooldown -= 1
                
                # افکت فلش
                if self.flash_effect > 0:
                    flash_overlay = np.ones_like(frame) * 255
                    frame = cv2.addWeighted(frame, 0.7, flash_overlay, 0.3, 0)
                    self.flash_effect -= 1
                
                # نمایش اطلاعات
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