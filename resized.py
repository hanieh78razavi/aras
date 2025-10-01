import cv2
import os

def batch_resize_images(input_folder, output_folder, new_width, new_height):
    """
    تغییر اندازه تمام تصاویر یک پوشه
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            # خواندن تصویر
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # تغییر اندازه
                resized_img = cv2.resize(img, (new_width, new_height))
                
                # ذخیره تصویر
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, resized_img)
                print(f"تصویر {filename} تغییر اندازه داده شد.")
            else:
                print(f"خطا در خواندن تصویر {filename}")

# استفاده مثال
input_folder = "input_images"
output_folder = "resized_images"
batch_resize_images(input_folder, output_folder, 800, 600) 