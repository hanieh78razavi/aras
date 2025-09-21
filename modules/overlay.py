import cv2
import numpy as np

def overlay_clothes(frame, keypoints, clothes_img, mp_pose):
    """
    قرار دادن لباس روی بدن
    frame: تصویر وبکم
    keypoints: لیست keypoints از MediaPipe
    clothes_img: تصویر PNG لباس با پس زمینه شفاف
    """
    # استفاده از شانه‌ها برای تعیین عرض لباس
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # محاسبه عرض و ارتفاع لباس (ساده)
    shoulder_dist = int(abs((left_shoulder.x - right_shoulder.x) * frame.shape[1]))
    width = max(shoulder_dist * 2, 1)
    height = int(width * clothes_img.shape[0] / clothes_img.shape[1])
    if width < 1 or height < 1:
        return frame

    # نقطه بالای لباس (وسط شانه‌ها)
    x_center = int(((left_shoulder.x + right_shoulder.x) / 2) * frame.shape[1])
    y_top = int(((left_shoulder.y + right_shoulder.y) / 2) * frame.shape[0])
    x1 = max(x_center - width // 2, 0)
    y1 = max(y_top, 0)
    x2 = min(x1 + width, frame.shape[1])
    y2 = min(y1 + height, frame.shape[0])

    # تغییر اندازه لباس
    resized_clothes = cv2.resize(clothes_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)

    # بررسی وجود کانال آلفا
    if resized_clothes.shape[2] != 4:
        return frame

    # ماسک آلفا
    alpha_clothes = resized_clothes[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_clothes

    # overlay لباس روی فریم (ایمن)
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha_clothes * resized_clothes[:, :, c] +
            alpha_frame * frame[y1:y2, x1:x2, c]
        ).astype(frame.dtype)

    return frame
