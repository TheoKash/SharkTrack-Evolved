'''
Script to convert selected tracks from a a SharkTrack YOLO CSV output to a 
VIAME-compatible CSV and extract corresponding frames as images.
Requires access to the original videos used.
'''

import os
import csv
import cv2
from collections import defaultdict


# ===== USER CONFIGURATION =====
annotations_csv = "/path/to/your/yolo/output.csv"  # SharkTrack YOLOv8 inference output CSV
output_dir = "/path/to/your/output/directory"      # Directory to save extracted images and VIAME CSV
selected_track_ids = []                            # List of track IDs to extract eg. [1, 2, 5, 10]
base_video_dir = "/path/to/videos"                 # Base directory containing the videos referenced in the YOLO output CSV
# ==============================

# Create output directories
output_images_dir = os.path.join(output_dir, "images") 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)
viame_output_csv = os.path.join(output_dir, "viame_annotations.csv")

# Data structures
frame_data = defaultdict(list)

# 1. Read and filter CSV annotations
with open(annotations_csv, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            track_id = int(row["track_id"])
            if track_id not in selected_track_ids:
                continue
            
            video_sub_path = row["track_metadata"].split("/")[0:-1]
            video_sub_path = "/".join(video_sub_path)
            video_path = os.path.join(base_video_dir, video_sub_path)
            frame_num = int(float(row["frame"]))
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            img_width = float(row["w"])
            img_height = float(row["h"])
            video_timestamp = row["time"]
            confidence = round(float(row["confidence"]), 3)
            
            # Store data grouped by (video_path, frame_num)
            key = (video_path, frame_num, video_timestamp)
            frame_data[key].append((xmin, ymin, xmax, ymax, img_width, img_height, track_id, confidence))
            
        except (KeyError, ValueError) as e:
            print(f"Skipping row due to error: {e}")
            continue

# Track existing files to avoid duplicates
existing_files = set()

viame_rows = []
image_id_counter = 1
seen_frame_keys = {}

# 2. Process videos and extract frames
for (video_path, frame_num, timestamp), bboxes in frame_data.items():
    # Generate unique filename base
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(video_path))
    filename_base = f"{parent_dir}_{video_name}_frame{frame_num:04d}"
    img_filename = f"{filename_base}.jpg"
    img_path = os.path.join(output_images_dir, img_filename)
    
    # Only process frame if we haven't created the image yet
    if img_path not in existing_files:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            exit()

        # Set video to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            print(f"Error reading frame {frame_num} from {video_path}")
            exit()

        # Save image
        cv2.imwrite(img_path, frame)
        print(f"Saved image: {img_path}")
        existing_files.add(img_path)  # Add to tracked files
    else:
        print(f"Skipping existing image: {img_path}")
        
     # Assign global image ID
    frame_key = (video_path, frame_num)
    if frame_key not in seen_frame_keys:
        seen_frame_keys[frame_key] = image_id_counter
        image_id_counter += 1
    image_id = seen_frame_keys[frame_key]

    # Build one VIAME row per bbox
    for (xmin, ymin, xmax, ymax, img_width, img_height, track_id, confidence) in bboxes:
        viame_rows.append([
            track_id,
            timestamp,
            image_id,
            round(xmin, 1),
            round(ymin, 1),
            round(xmax, 1),
            round(ymax, 1),
            1.0,                # detection confidence
            -1,                 # target length
            "No_species_data",  # species label
            1.0                 # species confidence
        ])
    
# 3. Write out VIAME CSV
with open(viame_output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "# 1: Track-id",
        "2: Timestamp",
        "3: Image Identifier",
        "4: TL_x",
        "5: TL_y",
        "6: BR_x",
        "7: BR_y",
        "8: Detection Confidence",
        "9: Target Length",
        "10: Species",
        "11: Species Confidence"
    ])
    for row in viame_rows:
        writer.writerow(row)

print(f"VIAME CSV written to: {viame_output_csv}")        
        