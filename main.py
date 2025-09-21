# main.py
# این فایل نقطه ورود برنامه است
# 1. وب‌کم رو باز می‌کنه
# 2. فریم‌ها رو می‌گیره
# 3. با ماژول‌ها (pose_detector, segmentation, overlay) پردازش می‌کنه
# 4. نمایش و ذخیره تصویر
import cv2
import mediapipe as mp
from modules.overlay import overlay_clothes

# راه‌اندازی MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# وب‌کم
cap = cv2.VideoCapture(0)

# بارگذاری لباس PNG (با alpha)

clothes_img = cv2.imread("assets/clothes/shirt1.png", cv2.IMREAD_UNCHANGED)
if clothes_img is None:
    raise FileNotFoundError("تصویر لباس پیدا نشد: assets/clothes/shirt1.png")
print("clothes_img.shape:", clothes_img.shape)
if clothes_img.shape[2] != 4:
    raise ValueError("تصویر لباس باید دارای کانال آلفا (RGBA) باشد. لطفاً یک PNG شفاف استفاده کنید.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # رسم اسکلت بدن
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # تبدیل landmarks به لیست ساده x,y
        keypoints = results.pose_landmarks.landmark

        # overlay لباس
    frame = overlay_clothes(frame, keypoints, clothes_img, mp_pose)

    cv2.imshow("Virtual Try-On", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
