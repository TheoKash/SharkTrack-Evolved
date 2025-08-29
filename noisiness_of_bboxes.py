'''
This script computes the Intersection Over Union and Average Center Distance 
between noisy and correct bounding boxes in YOLO format.
Requires each dataset to have a corresponding directory with correct labels 
ending in '_good_bboxes'.

OPTIONALLY:
1) Creates a new dataset including only images with bounding 
boxes above a specified IoU threshold. Outputs to a new directory ending in 
'_high_iou'.

2) Creates visualisations of noisy vs GT bboxes.


Directory structure:
BASE_DIR/
    data_from_site1/                <------ noisy labels
        images/
        labels/
    data_from_site1_good_bboxes/    <------ correct labels 
        images/
        labels/
    data_from_site2/
    .
    .
    .
'''

import os
import glob
import math
import shutil
from PIL import Image, ImageDraw

# ==== User-configurable parameters ====
BASE_DIR = '/path/to/datasets/directory'  # Base directory containing subdirectories of images and labels

                                          # e.g., dirs = ['B07_240', 'C05_120'] or dirs = []
dirs = []                                 # List of subdirectories to process; if empty, all subdirs will be processed
MIN_IOU = 0.5                             # IoU threshold to match bounding boxes
visualise_noisy_vs_ground_truth = True    # If True, will create visualisations of noisy vs GT bboxes
filter_low_IoU = False                    # If True, will create a new dataset including only images with bboxes above MIN_IOU
# =====================================

if not dirs:
    dirs = os.listdir(BASE_DIR)
dirs = [d for d in dirs if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.endswith('_good_bboxes')]


def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h
    
    w_abs = width * img_w
    h_abs = height * img_h
    
    x1 = x_center_abs - w_abs / 2
    y1 = y_center_abs - h_abs / 2
    x2 = x_center_abs + w_abs / 2
    y2 = y_center_abs + h_abs / 2
    return x1, y1, x2, y2, x_center_abs, y_center_abs


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0


def parse_yolo_labels(label_path, img_w, img_h):
    entries = []
    with open(label_path, 'r') as f:
        for line in f:
            
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Malformed line in label: {label_path}. Found {line.strip()}")
                continue
            
            xc, yc, w, h = map(float, parts[1:])
            x1, y1, x2, y2, cx, cy = yolo_to_bbox(xc, yc, w, h, img_w, img_h)
            entries.append(((x1, y1, x2, y2), (cx, cy)))
            
    return entries


def visualise_bboxes_vs_gt(img_path, corr_lbl, noisy_lbl, out_dir, fname, iou):
    """
    visualise ALL GT bboxes (red) and ALL noisy bboxes (blue).
    """
    os.makedirs(out_dir, exist_ok=True)
    with Image.open(img_path) as img:
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        # Draw GT (correct) bboxes in red
        for corr_box, _ in parse_yolo_labels(corr_lbl, img_w, img_h):
            draw.rectangle(corr_box, outline="red", width=3)

        # Draw noisy bboxes in blue
        if os.path.exists(noisy_lbl):
            for noisy_box, _ in parse_yolo_labels(noisy_lbl, img_w, img_h):
                draw.rectangle(noisy_box, outline="blue", width=3)

        # Label IoU
        if iou > 0:
            draw.text((10, 10), f"Best IoU={iou:.3f}", fill="yellow")
        else:
            draw.text((10, 10), "No overlap", fill="yellow")

        out_path = os.path.join(out_dir, f"{fname}_IoU_{iou:.3f}.jpg")
        img.save(out_path)


def compute_and_filter(noisy_dir, correct_dir, high_iou_dir):
    noisy_lbl_dir = os.path.join(noisy_dir, 'labels')
    noisy_img_dir = os.path.join(noisy_dir, 'images')
    correct_lbl_dir = os.path.join(correct_dir, 'labels')
    correct_img_dir = os.path.join(correct_dir, 'images')
    viz_dir = os.path.join(noisy_dir, 'visualised_bboxes')

    # prepare high_iou dirs only if filtering enabled
    if filter_low_IoU:
        high_img_dir = os.path.join(high_iou_dir, 'images')
        high_lbl_dir = os.path.join(high_iou_dir, 'labels')
        os.makedirs(high_img_dir, exist_ok=True)
        os.makedirs(high_lbl_dir, exist_ok=True)

    matched_ious = []
    matched_dists = []
    matched_norm_dists = []
    skipped_nonzero = []
    skipped_zero = 0
    missing = 0
    skipped_frames = set()

    for correct_lbl in glob.glob(os.path.join(correct_lbl_dir, '*.txt')):
        fname = os.path.basename(correct_lbl)
        noisy_lbl = os.path.join(noisy_lbl_dir, fname)
        if not os.path.exists(noisy_lbl):
            missing += 1
            continue

        img_name = fname.rsplit('.txt', 1)[0]
        img_path = os.path.join(correct_img_dir, img_name + '.jpg')
        noisy_img = os.path.join(noisy_img_dir, img_name + '.jpg')
        if not os.path.exists(img_path) or not os.path.exists(noisy_img):
            missing += 1
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size
        diag = math.hypot(img_w, img_h)

        correct_entries = parse_yolo_labels(correct_lbl, img_w, img_h)
        noisy_entries = parse_yolo_labels(noisy_lbl, img_w, img_h)
        if not noisy_entries:
            missing += len(correct_entries)
            continue

        for corr_box, (corr_cx, corr_cy) in correct_entries:
            best_iou = 0
            best_nx = best_ny = 0
            for nb_box, (nb_cx, nb_cy) in noisy_entries:
                iou = compute_iou(corr_box, nb_box)
                if iou > best_iou:
                    best_iou, best_nx, best_ny = iou, nb_cx, nb_cy

            if best_iou >= MIN_IOU:
                matched_ious.append(best_iou)
                dist = math.hypot(corr_cx - best_nx, corr_cy - best_ny)
                matched_dists.append(dist)
                matched_norm_dists.append(dist / diag)
            else:
                skipped_frames.add(fname)
                if visualise_noisy_vs_ground_truth:
                    visualise_bboxes_vs_gt(img_path, correct_lbl, noisy_lbl, viz_dir, img_name, best_iou)
                if best_iou > 0:
                    skipped_nonzero.append(best_iou)
                else:
                    skipped_zero += 1

        # if any bbox matched above MIN_IOU -> copy frame (only if filtering enabled)
        if filter_low_IoU and fname not in skipped_frames:
            shutil.copy2(noisy_img, os.path.join(high_img_dir, os.path.basename(noisy_img)))
            shutil.copy2(noisy_lbl, os.path.join(high_lbl_dir, os.path.basename(noisy_lbl)))

    matched = len(matched_ious)
    skipped = missing + len(skipped_nonzero) + skipped_zero
    avg_iou = sum(matched_ious) / matched if matched else 0
    avg_dist_px = sum(matched_dists) / matched if matched else 0
    avg_dist_norm = sum(matched_norm_dists) / matched if matched else 0
    avg_skip_iou = sum(skipped_nonzero) / len(skipped_nonzero) if skipped_nonzero else 0
    return matched, skipped, avg_iou, avg_dist_px, avg_dist_norm, avg_skip_iou, skipped_zero


def main():
    for d in sorted(dirs):
        noisy_dir = os.path.join(BASE_DIR, d)
        correct_dir = noisy_dir + '_good_bboxes'
        high_iou_dir = noisy_dir + '_high_iou'
        if not os.path.isdir(correct_dir):
            continue

        m, s, avg_iou, avg_px, avg_norm, avg_skip_iou, zero = compute_and_filter(noisy_dir, correct_dir, high_iou_dir)
        print(f"{d}: matched={m}, skipped={s} "
              f"(nonzero_avg_skip_iou={avg_skip_iou:.4f}, zero_count={zero}), "
              f"avg_IoU={avg_iou:.4f}, avg_dist={avg_px:.1f}px ({avg_norm:.3f} diag)")
        if filter_low_IoU:
            print(f" -> high IoU dataset written to: {high_iou_dir}")


if __name__ == '__main__':
    main()
