import os
import csv
import shutil
import cv2
from collections import defaultdict

'''
This script is for converting VIAME CSV annotations into YOLO format training data.
It copies images from a source directory to an output directory and creates 
corresponding YOLO label files from the viame .csv annotations.
'''

# ===== USER CONFIGURATION =====
viame_csv      = "/path/to/your/viame/annotations.csv"  # Input VIAME CSV file
src_images_dir = "/path/to/your/source/images"          # Directory containing the images
output_dir     = "/path/to/your/output/directory"       # Directory to save YOLO formatted data
# ==============================

# 1. Read VIAME CSV grouping by image basename
detections = defaultdict(list)
with open(viame_csv, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        # [track_id, image_basename, image_id, xmin, ymin, xmax, ymax, ...]
        _, basename, _, xmin, ymin, xmax, ymax, *_ = row
        detections[basename].append((float(xmin), float(ymin), float(xmax), float(ymax)))

# 2. Prepare YOLO‑style output dirs
yolo_img_dir   = os.path.join(output_dir, "images")
yolo_label_dir = os.path.join(output_dir, "labels")
os.makedirs(yolo_img_dir,   exist_ok=True)
os.makedirs(yolo_label_dir, exist_ok=True)

# 3. Copy images & write labels
for basename, bboxes in detections.items():
    src_jpg = basename
    src_path = os.path.join(src_images_dir, src_jpg)
    if not os.path.isfile(src_path):
        print(f"Warning: image not found: {src_path}")
        continue

    # only copy if not already present
    dst_img = os.path.join(yolo_img_dir, src_jpg)
    if not os.path.exists(dst_img):
        shutil.copy2(src_path, dst_img)
    else:
        print(f"Skipping copy, already exists: {dst_img}")

    # load to get dimensions
    img = cv2.imread(src_path)
    h, w = img.shape[:2]

    # write label file (overwrite or create)
    # remove the image extension (.jpg) from the basename
    basename_no_ext = os.path.splitext(basename)[0]
    lbl_path = os.path.join(yolo_label_dir, basename_no_ext + ".txt")
    with open(lbl_path, "w") as f:
        for xmin, ymin, xmax, ymax in bboxes:
            x_c = (xmin + xmax) / 2.0 / w
            y_c = (ymin + ymax) / 2.0 / h
            bw  = (xmax - xmin) / w
            bh  = (ymax - ymin) / h
            f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Processed {src_jpg}: {len(bboxes)} box(es)")

print("Done! YOLO data at:", output_dir)
