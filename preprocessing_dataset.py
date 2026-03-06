import cv2
import os
import numpy as np

CROP_CONFIGS = {
    "Tolong": (300, 600, 800),
    "TerimaKasih": (150, 600, 800),
    "Maaf": (300, 550, 800),
    "SamaSama": (150, 1200, 800)
}

def crop_and_resize_by_class(image_path, class_name, target_size=256, to_grayscale=False):

    # 1. Read image (OpenCV reads in BGR format)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image {image_path} could not be read!")
        return None
        
    h, w = img.shape[:2]
    
    if class_name not in CROP_CONFIGS:
        print(f"Class {class_name} not recognized!")
        return None
        
    start_x, start_y, box_size = CROP_CONFIGS[class_name]
    
    end_x = min(start_x + box_size, w)
    end_y = min(start_y + box_size, h)
    
    # 2. Crop image
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    crop_h, crop_w = cropped_img.shape[:2]
    
    # Padding if crop is not square
    if crop_h != crop_w:
        max_side = max(crop_h, crop_w)
        canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        canvas[0:crop_h, 0:crop_w] = cropped_img
        cropped_img = canvas
    
    # 3. Resize image
    resized_img = cv2.resize(cropped_img, (target_size, target_size))
    
    # 4. Optional grayscale conversion
    if to_grayscale:
        final_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    else:
        final_img = resized_img
    
    return final_img


def process_folder(input_folder, output_folder, class_name):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"crop_{filename}")

            processed_image = crop_and_resize_by_class(
                input_path,
                class_name,
                target_size=256
            )

            if processed_image is not None:
                
                cv2.imwrite(output_path, processed_image)
                print(f"Successfully processed: {filename}")

# USAGE
process_folder("dataset/Maaf", "dataset_preprocessing/Maaf", "Maaf")

