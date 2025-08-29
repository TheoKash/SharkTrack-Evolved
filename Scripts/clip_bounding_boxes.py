'''
Script to clip YOLO bounding box labels to ensure they stay within image boundaries.
Assumes labels are in YOLO format: <class> <x_center> <y_center> <width> <height>

This is needed because annotated labels can sometimes extend beyond image edges,
which can cause issues during training or evaluation.
'''

import os

basedir = '/path/to/your/dataset'  # Change this to your dataset path
label_dir = os.path.join(basedir, 'labels')

def clip(val):
    return max(0.0, min(val, 1.0))

for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue

    path = os.path.join(label_dir, label_file)
    new_lines = []

    with open(path, 'r') as f:
        
        for line in f:
                
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipped malformed line in {label_file}: {line.strip()}")
                continue  

            cls, xc, yc, w, h = map(float, parts)
            
            # Convert to box corners
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            # Clip to [0, 1]
            x1 = clip(x1)
            y1 = clip(y1)
            x2 = clip(x2)
            y2 = clip(y2)

            # Convert back to YOLO format
            new_xc = (x1 + x2) / 2
            new_yc = (y1 + y2) / 2
            new_w = x2 - x1
            new_h = y2 - y1

            # Skip box if it collapsed
            if new_w <= 0 or new_h <= 0:
                print(f"Skipped collapsed box in {label_file}: {line.strip()}")
                continue

            new_line = f"{int(cls)} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
            new_lines.append(new_line)
            print(f"✅ Processed {label_file}: {new_line}")

    # Overwrite file with cleaned labels
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
