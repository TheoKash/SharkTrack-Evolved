import csv
import os
from collections import defaultdict

# Set the label directory
label_dir = '/vol/biomedic3/bglocker/ugproj/tk1420/NorthSea1_val/val/labels'
os.makedirs(label_dir, exist_ok=True)

# Initialize a dictionary to store bounding box lines per frame_id
labels_dict = defaultdict(list)

# Image dimensions
img_width = 1920.0
img_height = 1080.0

annotations_path = '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Videos/NorthSea1/annotations.csv'
# Read and process the CSV file
with open(annotations_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            frame_id = int(row['frame_id'])
            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])
        except (ValueError, KeyError) as e:
            print(f"Skipping row due to error: {e}")
            continue

        # Calculate normalized values
        x_center = (xmin + xmax) / (2 * img_width)
        y_center = (ymin + ymax) / (2 * img_height)
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Format the bounding box line
        bbox_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        labels_dict[frame_id].append(bbox_line)

# Write each frame's bounding boxes to the corresponding label file
for frame_id, bbox_lines in labels_dict.items():
    filename = f"{frame_id:04d}.txt"
    filepath = os.path.join(label_dir, filename)
    
    # Check if the file already exists, and if so, append to it
    if os.path.exists(filepath):
        print('File already exists, appending to it:', filepath)
        with open(filepath, 'a') as f:
            for line in bbox_lines:
                f.write(line + '\n')
    else:
        with open(filepath, 'w') as f:
            for line in bbox_lines:
                f.write(line + '\n')

print("Label files generated successfully.")