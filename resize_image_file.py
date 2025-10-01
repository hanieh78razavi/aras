import cv2
import os
import math

ROOT = os.path.dirname(__file__)
PICTURES_DIR = os.path.join(ROOT, 'assets', 'pictures')
OUT_DIR = os.path.join(ROOT, 'captured_images')
os.makedirs(OUT_DIR, exist_ok=True)

# Preset sizes
PRESETS = {
    '1': (320, 240),
    '2': (640, 480),
    '3': (1280, 720),
    '4': (1920, 1080),
}

# Safety cap for upscaling (display & save)
MAX_W = 1920
MAX_H = 1080

# Gather images
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
files = [f for f in sorted(os.listdir(PICTURES_DIR)) if f.lower().endswith(EXTS)]
if not files:
    print('هیچ تصویری در پوشه assets/pictures یافت نشد.')
    raise SystemExit(1)

index = 0
scale = 1.0  # relative scale multiplier applied to original

window = 'Image Resizer'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

print('کلیدها:')
print('  n/p: تصویر بعدی/قبلی')
print('  1/2/3/4: انتخاب اندازهٔ پیش‌فرض (320x240,640x480,1280x720,1920x1080)')
print('  +/-: افزایش/کاهش اندازه به‌صورت درصدی (10%)')
print("  o: ذخیره روی همان فایل (overwrite)")
print("  a: ذخیره با نام جدید در پوشه 'captured_images'")
print('  q: خروج')


def load_image(idx):
    path = os.path.join(PICTURES_DIR, files[idx])
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"ناتوان در خواندن تصویر: {path}")
    return img, path


def clamp_size(w, h):
    w = max(1, int(w))
    h = max(1, int(h))
    if w > MAX_W or h > MAX_H:
        # scale down to fit MAX while keeping aspect
        aspect = w / h
        if w > MAX_W:
            w = MAX_W
            h = int(round(w / aspect))
        if h > MAX_H:
            h = MAX_H
            w = int(round(h * aspect))
    return w, h


orig_img, orig_path = load_image(index)
orig_h, orig_w = orig_img.shape[:2]

while True:
    # compute target size from presets or scale
    target_w = int(round(orig_w * scale))
    target_h = int(round(orig_h * scale))
    target_w, target_h = clamp_size(target_w, target_h)

    resized = cv2.resize(orig_img, (target_w, target_h), interpolation=cv2.INTER_AREA if scale<=1.0 else cv2.INTER_LINEAR)

    # overlay info
    info_lines = [
        f"file: {os.path.basename(orig_path)}",
        f"orig: {orig_w}x{orig_h}  ->  {target_w}x{target_h}  (scale={scale:.2f})",
        "keys: n/p next/prev | 1-4 presets | +/- scale | o overwrite | a save-as | q quit",
    ]
    y = 20
    for line in info_lines:
        cv2.putText(resized, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 25

    cv2.imshow(window, resized)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        index = (index + 1) % len(files)
        orig_img, orig_path = load_image(index)
        orig_h, orig_w = orig_img.shape[:2]
        scale = 1.0
    elif key == ord('p'):
        index = (index - 1) % len(files)
        orig_img, orig_path = load_image(index)
        orig_h, orig_w = orig_img.shape[:2]
        scale = 1.0
    elif chr(key) in PRESETS:
        w, h = PRESETS[chr(key)]
        # compute scale that maps original to preset while keeping aspect
        scale_w = w / orig_w
        scale_h = h / orig_h
        # choose scale that fits the preset while keeping aspect (use min)
        scale = min(scale_w, scale_h)
    elif key in (ord('+'), ord('=')):
        scale *= 1.1
        # avoid insane upscales
        if orig_w * scale > MAX_W or orig_h * scale > MAX_H:
            scale = min(MAX_W / orig_w, MAX_H / orig_h)
    elif key == ord('-'):
        scale /= 1.1
        if scale < 0.1:
            scale = 0.1
    elif key == ord('o'):
        # overwrite original file
        try:
            cv2.imwrite(orig_path, resized)
            print(f"فایل بازنویسی شد: {orig_path}")
            # reload original to reflect changes
            orig_img, orig_path = load_image(index)
            orig_h, orig_w = orig_img.shape[:2]
            scale = 1.0
        except Exception as e:
            print(f"خطا در ذخیره روی همان فایل: {e}")
    elif key == ord('a'):
        # save as new file in OUT_DIR
        base, ext = os.path.splitext(os.path.basename(orig_path))
        out_name = f"{base}_resized{ext}"
        out_path = os.path.join(OUT_DIR, out_name)
        # ensure unique
        i = 1
        while os.path.exists(out_path):
            out_name = f"{base}_resized_{i}{ext}"
            out_path = os.path.join(OUT_DIR, out_name)
            i += 1
        try:
            cv2.imwrite(out_path, resized)
            print(f"فایل ذخیره شد: {out_path}")
        except Exception as e:
            print(f"خطا در ذخیره: {e}")
    else:
        print(f"کلید ناشناخته: {key}")

cv2.destroyAllWindows()
