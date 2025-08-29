'''
Script to draw bounding boxes on images based on YOLO format labels.

Directory structure:
- BASE_DIR/
    - images/      # contains original images (e.g. *.jpg)
    - labels/      # contains YOLO format labels (e.g. *.txt)
    - images_annotated/  # output directory for images with drawn boxes
'''

import os
import cv2

# ===== USER CONFIGURATION =====
BASE_DIR = '/path/to/your/dataset'  # Change this to your dataset path
# ==============================

OUTPUT_DIR = os.path.join(BASE_DIR, 'images_annotated')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
LABEL_DIR = os.path.join(BASE_DIR, 'labels')
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(image_file)[0] + '.txt')
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        continue

    height, width = image.shape[:2]

    drawn = False

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Skipped malformed line in {label_path}: {line.strip()}")
                    continue

                class_id, x_center, y_center, w, h = map(float, parts)

                x_center *= width
                y_center *= height
                w *= width
                h *= height

                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Clamp coordinates
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, str(int(class_id)), (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    drawn = True
                else:
                    print(f"Invalid box in {label_path}: {x1}, {y1}, {x2}, {y2}")

    if not drawn:
        print(f"No boxes drawn for {image_file}")

    output_path = os.path.join(OUTPUT_DIR, image_file)
    cv2.imwrite(output_path, image)

print("✅ Annotation complete.")
