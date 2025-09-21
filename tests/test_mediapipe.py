
import sys
import time

try:
    import cv2
except ImportError:
    print("خطا: OpenCV نصب نشده است.\nدستور نصب: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("خطا: mediapipe نصب نشده است.\nدستور نصب: pip install mediapipe")
    sys.exit(1)

def main(camera_index=0):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"وب‌کم با شماره {camera_index} پیدا نشد!")
        return

    print("برای خروج کلید ESC را فشار دهید.")
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("فریم دریافت نشد!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # محاسبه و نمایش FPS
        frame_count += 1
        if frame_count >= 10:
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("MediaPipe Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # امکان انتخاب وب‌کم از خط فرمان
    idx = 0
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1])
        except ValueError:
            print("شماره وب‌کم باید عدد باشد.")
            sys.exit(1)
    main(idx)
