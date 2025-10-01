import cv2
import os
import time
import threading
import queue
from datetime import datetime

save_folder = 'captured_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Try to open the default camera (0). If you have multiple cameras, change the index.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("خطا: نمی‌توان به وب‌کم دسترسی پیدا کرد.")
    raise SystemExit(1)

# We will show the "photo saved" text for a short duration after a capture
overlay_duration = 0.5  # seconds
last_saved_time = 0
last_saved_name = ''
saved_count = 0
# Track previous space key state to avoid repeated saves when holding the key
last_space_pressed = False

# Background writer queue to avoid blocking the main loop when saving images
save_queue = queue.Queue()

# Performance: optionally reduce resolution to speed up preview and saves.
# Toggle at runtime with 't'. Default: True to improve FPS on slower machines.
reduce_resolution = True
low_res = (640, 480)


def _writer_thread_fn(q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        filename, img = item
        try:
            cv2.imwrite(filename, img)
        except Exception as e:
            print(f"خطا در ذخیره فایل {filename}: {e}")
        finally:
            q.task_done()


_writer_thread = threading.Thread(target=_writer_thread_fn, args=(save_queue,), daemon=True)
_writer_thread.start()

print("دستورالعمل:")
print("برای گرفتن عکس، کلید فاصله را فشار دهید.")
print("برای خروج، کلید 'q' را فشار دهید.")

window_name = 'Webcam - press space to capture'
cv2.namedWindow(window_name)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("خطا: فریم از وب‌کم خوانده نشد.")
            break

        # If a photo was saved recently, overlay a brief message
        if time.time() - last_saved_time < overlay_duration:
            text = 'تصویر ذخیره شد' if not last_saved_name else f'ذخیره شد: {os.path.basename(last_saved_name)}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Optionally resize to low resolution to improve speed
        if reduce_resolution and low_res:
            frame = cv2.resize(frame, low_res)

        cv2.imshow(window_name, frame)

        # Wait a short time to keep the loop responsive. Use 30 ms (~33 FPS).
        key = cv2.waitKey(30) & 0xFF

        # Determine current space key state (True if pressed)
        space_pressed = (key == 32)


        # Exit on 'q'
        if key == ord('q'):
            print('خروج...')
            break

        # On rising edge of space key, save image (prevents multiple saves when holding)
        if space_pressed and not last_space_pressed:
            # Create a unique filename using timestamp to avoid overwriting
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            photo_filename = os.path.join(save_folder, f'photo_{timestamp}.jpg')
            try:
                save_queue.put((photo_filename, frame.copy()))
                last_saved_time = time.time()
                last_saved_name = photo_filename
                saved_count += 1
                print(f"عکس ذخیره شد (queued): {photo_filename}")
            except Exception as e:
                print(f'خطا: افزودن به صف ذخیره انجام نشد: {e}')

        # Toggle low resolution mode with 't'
        if key == ord('t'):
            reduce_resolution = not reduce_resolution
            print(f"رزولوشن پایین {'فعال' if reduce_resolution else 'غیرفعال'} شد")

        # Update last space state for edge detection
        last_space_pressed = space_pressed

        # If a photo was saved recently, optionally show the count too
        if time.time() - last_saved_time < overlay_duration and last_saved_name:
            text = f'ذخیره شد: {os.path.basename(last_saved_name)} ({saved_count})'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
finally:
    # Ensure resources are always released
    cap.release()
    # stop writer thread
    try:
        save_queue.put(None)
        _writer_thread.join(timeout=2.0)
    except Exception:
        pass
    cv2.destroyAllWindows()
        