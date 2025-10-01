import cv2
import mediapipe as mp

# 1. مقداردهی اولیه MediaPipe
# ماژول های مورد نیاز برای تشخیص وضعیت و ابزارهای ترسیم را فراخوانی می کنیم.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 2. تنظیم مدل MediaPipe PoseLandmarker
# ما از حالت IMAGE (تصویر ثابت) برای این تمرین استفاده می کنیم.
# min_detection_confidence: حداقل اطمینان برای تشخیص موفقیت آمیز فرد
pose_detector = mp_pose.Pose(
    static_image_mode=True, 
    min_detection_confidence=0.5, 
    model_complexity=1
) 
# در نسخه های جدیدتر ممکن است از PoseLandmarker به جای mp_pose.Pose استفاده شود [1]

# 3. بارگذاری و پردازش تصویر ورودی
# مسیر تصویر خود را اینجا وارد کنید
IMAGE_PATH = 'sample_image.jpg' 

# خواندن تصویر با استفاده از OpenCV
# OpenCV به صورت پیش فرض از فرمت BGR استفاده می کند.
image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    print(f"خطا: تصویر {IMAGE_PATH} یافت نشد. لطفا مطمئن شوید که فایل در مسیر صحیح است.")
else:
    # 4. تبدیل تصویر به فرمت RGB
    # MediaPipe به فرمت RGB نیاز دارد.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 5. اجرای مدل تشخیص وضعیت
    # متد process() لندمارک های 33 گانه را استخراج می کند.
    results = pose_detector.process(image_rgb)

    # 6. ترسیم لندمارک ها و اسکلت بدن
    # اگر لندمارکی تشخیص داده شده باشد (نتایج خالی نباشد):
    if results.pose_landmarks:
        # ترسیم لندمارک ها و اتصالات اسکلت روی تصویر BGR اصلی
        mp_drawing.draw_landmarks(
            image_bgr, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS, # استفاده از اتصالات استاندارد MediaPipe
            # تنظیم ظاهر نقطه ها (قرمز با دایره کوچک)
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), 
            # تنظیم ظاهر خطوط (سبز)
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        # اطلاعات کلیدی خروجی
        # مختصات هر لندمارک (x، y، z و قابلیت دید) در نتایج ذخیره شده است [[3]، S_S7].
        # برای مثال، می توانیم مختصات بینی (لندمارک 0) را چاپ کنیم:
        nose_landmark = results.pose_landmarks.landmark
        print(f"مختصات بینی (x, y, visibility): ({nose_landmark.x:.2f}, {nose_landmark.y:.2f}, {nose_landmark.visibility:.2f})")
        print(f"مختصات عمق نسبی (z) بینی: {nose_landmark.z:.2f}")


    # 7. نمایش تصویر خروجی
    cv2.imshow('MediaPipe Pose Detection', image_bgr)
    cv2.waitKey(0) # صبر می کند تا کلیدی فشرده شود
    cv2.destroyAllWindows()