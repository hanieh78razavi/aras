import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List, Any

# مقداردهی اولیه MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class PoseDetector:
    """کلاس اصلی برای تشخیص پوز با دکمه لمسی مجازی"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        
        self.pose_detector = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # اضافه کردن مدل تشخیص دست برای انگشت
        self.hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.cap = None
        self.button_cooldown = 0  # جلوگیری از عکس‌های پشت سر هم

    def get_persian_font(self) -> Optional[str]:
        """پیدا کردن فونت فارسی ساده"""
        persian_fonts = [
            'Vazir.ttf', 'Vazir-Regular.ttf', 'Vazir-FD.ttf',
            'B Nazanin.ttf', 'B Yekan.ttf',
            'IRANSans.ttf', 'IRANSansWeb.ttf',
            'Shabnam.ttf', 'Shabnam-FD.ttf',
            'Tahoma.ttf', 'Arial.ttf'
        ]
        
        font_paths = [
            'C:/Windows/Fonts/',
            os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts/'),
            './fonts/'
        ]
        
        current_user = os.getenv('USERNAME') or os.getenv('USER')
        if current_user:
            user_font_path = f'C:/Users/{current_user}/AppData/Local/Microsoft/Windows/Fonts/'
            font_paths.append(user_font_path)
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    available_fonts = os.listdir(font_path)
                    for font in persian_fonts:
                        if font in available_fonts:
                            return os.path.join(font_path, font)
                except (PermissionError, FileNotFoundError):
                    continue
        
        return None

    def put_persian_text(self, img: np.ndarray, text: str, position: Tuple[int, int], 
                        font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """نمایش متن فارسی روی تصویر"""
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            font_path = self.get_persian_font()
            font_size = max(20, int(28 * font_scale))
            
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = None
            
            if font:
                draw.text(position, bidi_text, font=font, fill=color)
            else:
                draw.text(position, bidi_text, fill=color)
            
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
        except ImportError:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return img
        except Exception:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return img

    def initialize_camera(self, camera_index: int = 0) -> bool:
        """راه‌اندازی دوربین"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                for i in range(1, 3):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                else:
                    return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("✅ Camera initialized - Press 'q' to quit")
            print("🖐️ Touch the virtual button with your finger to take photo!")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[Any, Any]:
        """پردازش فریم برای تشخیص پوز و دست"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # تشخیص پوز
            pose_results = self.pose_detector.process(image_rgb)
            
            # تشخیص دست
            hand_results = self.hand_detector.process(image_rgb)
            
            return pose_results, hand_results
        except Exception:
            return None, None

    def draw_landmarks(self, frame: np.ndarray, pose_results: Any, hand_results: Any) -> np.ndarray:
        """رسم اسکلت بدن و نقاط دست"""
        # رسم اسکلت بدن
        if pose_results and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # رسم نقاط دست (اختیاری - برای دیباگ)
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

    def is_finger_touching_button(self, hand_results: Any, frame_shape: Tuple[int, int], 
                                button_pos: Tuple[int, int, int, int]) -> bool:
        """بررسی آیا انگشت کاربر روی دکمه است"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return False
        
        h, w = frame_shape[:2]
        button_x1, button_y1, button_x2, button_y2 = button_pos
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # گرفتن موقعیت نوک انگشت اشاره (landmark 8)
            fingertip = hand_landmarks.landmark[8]
            
            # تبدیل مختصات نسبی به پیکسل
            fingertip_x = int(fingertip.x * w)
            fingertip_y = int(fingertip.y * h)
            
            # بررسی برخورد با دکمه
            if (button_x1 <= fingertip_x <= button_x2 and 
                button_y1 <= fingertip_y <= button_y2):
                return True
        
        return False

    def draw_virtual_button(self, frame: np.ndarray, is_touching: bool) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """رسم دکمه مجازی و برگرداندن موقعیت آن"""
        h, w = frame.shape[:2]
        
        # موقعیت دکمه - گوشه بالا راست
        button_width, button_height = 120, 60
        button_x1, button_y1 = w - button_width - 20, 20
        button_x2, button_y2 = button_x1 + button_width, button_y1 + button_height
        
        # رنگ دکمه بر اساس حالت
        button_color = (0, 200, 0) if is_touching else (200, 200, 200)  # سبز هنگام لمس
        text_color = (255, 255, 255) if is_touching else (0, 0, 0)
        
        # رسم دکمه
        cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), button_color, -1)
        cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (0, 0, 0), 2)
        
        # متن دکمه
        button_text = "عکس 📸" if is_touching else "عکس"
        text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = button_x1 + (button_width - text_size[0]) // 2
        text_y = button_y1 + (button_height + text_size[1]) // 2
        
        cv2.putText(frame, button_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return frame, (button_x1, button_y1, button_x2, button_y2)

    def take_screenshot(self, frame: np.ndarray) -> str:
        """ذخیره عکس از فریم فعلی"""
        try:
            timestamp = cv2.getTickCount()
            filename = f"touch_capture_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            print(f"✅ عکس ذخیره شد: {filename}")
            return filename
        except Exception as e:
            print(f"❌ خطا در ذخیره عکس: {e}")
            return ""

    def display_minimal_ui(self, frame: np.ndarray, pose_results: Any, 
                         hand_results: Any, button_pos: Tuple[int, int, int, int]) -> np.ndarray:
        """نمایش رابط کاربری مینیمال با دکمه لمسی"""
        h, w = frame.shape[:2]
        
        # وضعیت تشخیص - گوشه بالا چپ
        status_text = "وضعیت: فعال" if pose_results and pose_results.pose_landmarks else "وضعیت: غیرفعال"
        status_color = (0, 100, 0) if pose_results and pose_results.pose_landmarks else (0, 0, 100)
        frame = self.put_persian_text(frame, status_text, (20, 20), 0.8, status_color)
        
        # نمایش راهنمای لمسی
        guide_text = "انگشت خود را روی دکمه ببرید"
        frame = self.put_persian_text(frame, guide_text, (20, h - 30), 0.6, (100, 100, 100))
        
        # بررسی لمس دکمه
        is_touching = self.is_finger_touching_button(hand_results, frame.shape, button_pos)
        
        # کاهش کول‌داون
        if self.button_cooldown > 0:
            self.button_cooldown -= 1
        
        # اگر انگشت روی دکمه است و کول‌داون تمام شده
        if is_touching and self.button_cooldown == 0:
            self.take_screenshot(frame)
            self.button_cooldown = 30  # 30 فریم کول‌داون (حدود 1 ثانیه)
        
        return frame, is_touching

    def run(self):
        """اجرای اصلی برنامه"""
        if not self.initialize_camera():
            return
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # پردازش فریم
                frame = cv2.flip(frame, 1)
                pose_results, hand_results = self.process_frame(frame)
                
                # موقعیت ثابت دکمه
                h, w = frame.shape[:2]
                button_pos = (w - 140, 20, w - 20, 80)  # x1, y1, x2, y2
                
                # رسم اسکلت بدن و دست
                frame = self.draw_landmarks(frame, pose_results, hand_results)
                
                # رسم دکمه مجازی
                is_touching = self.is_finger_touching_button(hand_results, frame.shape, button_pos)
                frame = self.draw_virtual_button(frame, is_touching)[0]
                
                # نمایش رابط کاربری
                frame, _ = self.display_minimal_ui(frame, pose_results, hand_results, button_pos)
                
                # نمایش فریم
                cv2.imshow('Pose Detection - Virtual Touch', frame)
                
                # خروج با کلید q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Program stopped by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """پاکسازی منابع"""
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()
            self.hand_detector.close()
            print("✅ Resources released")
        except:
            pass

def main():
    """تابع اصلی"""
    print("🚀 Starting Pose Detection with Virtual Touch...")
    
    try:
        detector = PoseDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Program error: {e}")
    finally:
        print("👋 Program ended")

if __name__ == "__main__":
    main()