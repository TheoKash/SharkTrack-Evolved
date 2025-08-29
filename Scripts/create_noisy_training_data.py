import os
import csv
import cv2
from collections import defaultdict

''' 
NOISY TRAINING DATA CREATION TOOL

Allows you to create training data (image/label pairs) from Sharktrack outputs.
Extracts annotations for given track-ids from a CSV file (eg output.csv from running Sharktrack), 
saves the images and corresponding labels in YOLO format, and creates validation videos with bounding boxes for manual validation.
It then waits for the user to delete any false tracks (bad images) and deletes the corresponding labels.
'''

# ===== USER CONFIGURATION =====
annotations_csv = "path/to/your/yolo/output.csv"  # SharkTrack YOLOv8 inference output CSV
output_dir = "path/to/your/output/directory"      # Directory to save images, labels, and validation videos
selected_track_ids = []                           # List of track IDs to extract eg. [1, 2, 5, 10]
base_video_dir = "path/to/videos"                 # Base directory containing the videos referenced in the YOLO output CSV
# ==============================

validation_fps = 3 # FPS for validation videos

# Create output directories
output_images_dir = os.path.join(output_dir, "images") 
output_labels_dir = os.path.join(output_dir, "labels")
output_validation_dir = os.path.join(output_dir, "validation_videos")
output_annotated_images_dir = os.path.join(output_dir, "annotated_images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs(output_validation_dir, exist_ok=True)
os.makedirs(output_annotated_images_dir, exist_ok=True) 

frame_data = defaultdict(list)
extracted_frames_info = []  # Store info for validation video creation

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

# 2. Process videos and extract frames
for (video_path, frame_num, timestamp), bboxes in frame_data.items():
    # Generate unique filename base
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(video_path))
    filename_base = f"{parent_dir}_{video_name}_frame{frame_num:04d}"
    img_filename = f"{filename_base}.jpg"
    img_path = os.path.join(output_images_dir, img_filename)
    label_path = os.path.join(output_labels_dir, f"{filename_base}.txt")
    
    # Only process frame if we haven't created the image yet
    if img_path not in existing_files:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            continue

        # Set video to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            print(f"Error reading frame {frame_num} from {video_path}")
            continue

        # Save image
        cv2.imwrite(img_path, frame)
        print(f"Saved image: {img_path}")
        existing_files.add(img_path)  # Add to tracked files
    else:
        print(f"Skipping existing image: {img_path}")

    # Create or append to label file
    with open(label_path, "a") as label_file:  # Changed to append mode
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, img_width, img_height, track_id, confidence = bbox
            
            # Calculate normalized values
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Write YOLO-formatted line
            label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            label_file.write(label_line)
    
    print(f"Updated label: {label_path}")
    
    # Store info for validation video (only once per frame)
    extracted_frames_info.append({
        'video_path': video_path,
        'frame_num': frame_num,
        'img_path': img_path,
        'label_path': label_path,
        'parent_dir': parent_dir,
        'video_name': video_name,
        'time': timestamp,
    })

print("Extraction completed! Creating validation videos...")

# ===== VALIDATION VIDEO CREATION =====
# Group frames by video for validation videos
video_groups = defaultdict(list)
for frame_info in extracted_frames_info:
    key = (frame_info['video_path'], frame_info['parent_dir'], frame_info['video_name'])
    video_groups[key].append(frame_info)

# Create a validation video for each video group
for (video_path, parent_dir, video_name), frames in video_groups.items():
    # Sort frames by frame number
    frames = sorted(frames, key=lambda x: x['frame_num'])
    
    # Remove duplicate frames (same frame number)
    unique_frames = {}
    for frame_info in frames:
        frame_num = frame_info['frame_num']
        if frame_num not in unique_frames:
            unique_frames[frame_num] = frame_info
    
    # Get video resolution from first frame
    first_frame_info = frames[0]
    first_frame = cv2.imread(first_frame_info['img_path'])
    if first_frame is None:
        print(f"Couldn't read first frame for {video_name}, skipping validation video")
        continue
        
    height, width, _ = first_frame.shape
    validation_video_path = os.path.join(
        output_validation_dir, 
        f"{parent_dir}_{video_name}_validation.mp4"
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(validation_video_path, fourcc, validation_fps, (width, height))
    
    print(f"Creating validation video: {validation_video_path}")
    print(f"  - Contains {len(unique_frames)} frames")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {validation_fps}")
    
    # Process each unique frame
    for frame_num, frame_info in unique_frames.items():
        frame = cv2.imread(frame_info['img_path'])
        if frame is None:
            continue
            
        # Draw bounding boxes from label file
        with open(frame_info['label_path'], 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    assert False, f"Invalid label format in {frame_info['label_path']} at line {i+1}: {line.strip()}"
                if parts[0] != '0':
                    print(f"Error: This script only exprects a single class right now, got {parts[0]} in {frame_info['label_path']} at line {i+1}")
                    assert False, f"Expected class ID '0' in {frame_info['label_path']} at line {i+1}, got {parts[0]}"
                    
                # Parse YOLO format
                _, x_center, y_center, w, h = map(float, parts)
                
                # Convert to image coordinates
                x_min = int((x_center - w/2) * width)
                y_min = int((y_center - h/2) * height)
                x_max = int((x_center + w/2) * width)
                y_max = int((y_center + h/2) * height)
                
                track_id = frame_data[(frame_info['video_path'], frame_info['frame_num'], frame_info['time'])][i][6] 
                
                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'Elasmobranch-{track_id}', (x_min, y_min-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Overlay
        overlay_font = (cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_info['frame_num']}", (10, 30), *overlay_font)
        cv2.putText(frame, f"Video: {video_name}", (10, 70), *overlay_font)
        
        # Add timestamp overlay 
        time_str = frame_info.get('time', 'N/A')
        cv2.putText(frame, f"Timestamp: {time_str}", (10, 110), *overlay_font)
        
        # Save annotated frame to new directory
        annotated_img_path = os.path.join(output_annotated_images_dir, os.path.basename(frame_info['img_path']))
        cv2.imwrite(annotated_img_path, frame)
        
        out.write(frame)
    
    out.release()
    print(f"  - Validation video saved: {validation_video_path}")

print("Processing completed!")
print(f"Extracted {len(extracted_frames_info)} frames")
print(f"Created {len(video_groups)} validation videos")

# Wait for user input before the next step
print("Do your manual validation - delete bad annotated images in the 'annotated_images' directory")
input("Press Enter to delete corresponding images and labels...")

# NEW CLEANUP LOGIC: Delete based on annotated images
deleted_count = 0
for frame_info in extracted_frames_info:
    # Get corresponding annotated image path
    annotated_img_path = os.path.join(output_annotated_images_dir, os.path.basename(frame_info['img_path']))
    
    # If annotated image was deleted, delete original image and label
    if not os.path.exists(annotated_img_path):
        # Delete original image
        if os.path.exists(frame_info['img_path']):
            os.remove(frame_info['img_path'])
            print(f"Deleted image: {frame_info['img_path']}")
        else:
            assert False, f"Image not found: {frame_info['img_path']}"
        
        # Delete label
        if os.path.exists(frame_info['label_path']):
            os.remove(frame_info['label_path'])
            print(f"Deleted label: {frame_info['label_path']}")
        else:
            assert False, f"Label not found: {frame_info['label_path']}"
        
        deleted_count += 1

print(f"Deleted {deleted_count} images and labels")

# Validate that all images have corresponding labels
missing_labels = []
missing_images = []
for frame_info in extracted_frames_info:
    if os.path.exists(frame_info['img_path']) and not os.path.exists(frame_info['label_path']):
        missing_labels.append(frame_info['img_path'])
    if os.path.exists(frame_info['label_path']) and not os.path.exists(frame_info['img_path']):
        missing_images.append(frame_info['label_path'])

if missing_labels:
    print(f"Warning: {len(missing_labels)} images have no corresponding labels:")
    for img in missing_labels:
        print(f"  - {img}")
if missing_images:
    print(f"Warning: {len(missing_images)} labels have no corresponding images:")
    for label in missing_images:
        print(f"  - {label}")

print("Validation and cleanup completed!")