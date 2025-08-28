from ultralytics import YOLO
import pandas as pd
import wandb
import os
import sys
from pathlib import Path
# from evaluation.evaluate_yolo_tracker import evaluate

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from evaluation.evaluate_yolo_tracker import evaluate_per_sequence

params = {
  # nvm this is cheating, I can only test with NS data. And I should test with the typical Alv4 val for ease of comp. (maybe?)
  'name': 'report-Alv4_v8n_augs_high_iou', # 'report-ALv2-v8n_augs_good_bbs', # 'report-ALv2-v8n_augs_good_bbs',
  # 'name': 'p3v6_n2',
  'model_size': 'n', # n, s, m, l, x
  # 'pretrained_model': "/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8n_1000e_mosaic0.8_perspective0.0005_cutmix0.15/weights/last.pt",
  # 'pretrained_model': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/old/shark_locator_tests/runs/detect/yolov8m_mvd2/best.pt',
  # 'pretrained_model': '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8s3/weights/last.pt',
  # 'pretrained_model': None,
  'pretrained_model': '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/report-Alv4_v8n_augs_high_iou/weights/last.pt',
  'epochs': 500, 
  'imgsz': 640,
  
  'patience': 150, #50
  
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/p3v6_train_with_NS_val',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/datasets/p3v6/data_config.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/p3v6_plus_NS1_train_val__plus_bgs/data_config.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/datasets/p3v6_plus_AL_ns_bgs/data_config.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv1.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv2.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv3.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv4.yaml',
  'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv4_high_iou.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv1_good_bbs.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv2_good_bbs.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv3_good_bbs.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv4_good_bbs.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/ALv4_good_bbs_tv_split.yaml',
  'project_folder': 'models',
  'batch': 64,
  "iou_association_threshold": 0.5,
  # "tracker": "botsort.yaml",
  "tracker": "tracker_5fps.yaml",
  "conf_threshold": 0.2,
  "yolo_version": 'yolov8',
  # "yolo_version": 'yolo11',
  # augmentations
  'augmentations': {
    'hsv_h': 0.015,           # hue jitter
    'hsv_s': 0.7,             # saturation jitter
    'hsv_v': 0.4,             # value jitter
    'degrees': 20,            # rotation
    'translate': 0.1,         # translation
    'scale': 0.6,             # scale
    'shear': 0.0,             # shear
    'perspective': 0.0005,    # perspective
    'flipud': 0.0,            # vertical flip
    'fliplr': 0.5,            # horizontal flip
    'bgr': 0.0,               # BGR - flips channels
    'mosaic': 0.8, #0.8,            # mosaic
    'mixup': 0.0,             # mixup
    # 'cutmix': 0.1,            # cutmix
  }
#   base_augmentations = {
#     'hsv_h': 0.015,           # hue jitter
#     'hsv_s': 0.7,             # saturation jitter
#     'hsv_v': 0.4,             # value jitter
#     'degrees': 0.0,           # rotation
#     'translate': 0.1,         # translation
#     'scale': 0.5,             # scale
#     'shear': 0.0,             # shear
#     'perspective': 0.0,       # perspective
#     'flipud': 0.0,            # vertical flip
#     'fliplr': 0.5,            # horizontal flip
#     'bgr': 0.0,               # BGR - flips channels
#     'mosaic': 1.0,            # mosaic
#     'mixup': 0.0,             # mixup
#     'cutmix': 0.0,            # cutmix
# }
  # ,
  # 'lighting_aug': {
  #   'highlight_boost': 0.85,
  #   'shadow_darken': 0.6,
  #   'saturation_reduction': 0.35
  # }
}

model = YOLO(params['pretrained_model'] or f"{params['yolo_version']}{params['model_size']}.pt")  # load a pretrained model (recommended for training)

##############################################################
# Harsh Lighting Augmentation
##############################################################

# import random
# import cv2
# import numpy as np

# def apply_harsh_lighting(
#     img,
#     highlight_boost = 0.85,
#     shadow_darken = 0.6,
#     saturation_reduction = 0.35,
#     lighting_type="random"  # "radial", "left", "right", or "random"
# ):
#     """Apply harsh lighting with radial or directional effects"""
#     # Convert to float32 for processing
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
#     h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
#     rows, cols = v.shape[:2]
    
#     # Randomly select lighting type if unspecified
#     if lighting_type == "random":
#         lighting_type = random.choice(["radial", "left", "right"])
#         print(f"Using random lighting type: {lighting_type}")
    
#     # Generate lighting mask
#     if lighting_type == "radial":
#         center_x, center_y = cols // 2, rows // 2
#         max_radius = np.sqrt(center_x**2 + center_y**2)
#         y, x = np.ogrid[:rows, :cols]
#         dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_radius
#         mask = 1 - np.clip(dist, 0, 1)
#     else:  # Directional (left/right)
#         grad = np.linspace(0, 1, cols)
#         if lighting_type == "left":
#             mask = 1 - grad  # Bright on left, dark on right
#         else:  # "right"
#             mask = grad     # Bright on right, dark on left
#         mask = np.tile(mask, (rows, 1))  # Expand to 2D

#     # Apply lighting effects
#     v = v * (1 + highlight_boost * mask)
#     v = np.clip(v, 0, 255)
    
#     v = v * (1 - shadow_darken * (1 - mask))
#     v = np.clip(v, 0, 255)
    
#     s = s * (1 - saturation_reduction * mask)
#     s = np.clip(s, 0, 255)

#     # Convert back to RGB
#     hsv_out = np.stack([h, s, v], axis=-1)
#     return cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2RGB)
        


# Train the model
model.train(
  # resuming
  resume=True,
  workers=6,
  data=params['data_yaml'],
  epochs=params['epochs'],
  imgsz=params['imgsz'],
  patience=params['patience'],
  name=params['name'],
  batch=params['batch'],
  project = params['project_folder'],
  verbose=True,
  save_period=50,
  **params['augmentations'],
)


# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v11_1000e_mosaic0.8_perspective0.0005_cutmix0.1/weights/best.pt')

# Get mAP
model_folder = os.path.join(params['project_folder'], params['name'])
model_folder = os.path.abspath(model_folder)
assert os.path.exists(model_folder), 'Model folder does not exist'
print(f'Model folder: {model_folder}')

results_path = os.path.join(model_folder, 'results.csv')
assert os.path.exists(results_path), 'Results file does not exist'

results = pd.read_csv(results_path)
results.columns = results.columns.str.strip()
best_mAP = results['metrics/mAP50(B)'].max()
print(f"Best mAP: {best_mAP}")
print("Exiting before tracking")
exit()


# track
model_path = os.path.join(model_folder, 'weights', 'best.pt')
assert os.path.exists(model_path), 'Model file does not exist'

model_name = os.path.basename(model_path).replace('.pt', '')
print(f"Tracking with model: {model_name}")
metrics_per_sequence, track_time, device, all_aligned_annotations = evaluate_per_sequence(model_path, params["conf_threshold"], iou_association_threshold=0.5,imgsz=640, tracker_type='tracker_5fps.yaml')
print(f"Per-sequence metrics for {model_name}:")
for seq, metrics in metrics_per_sequence.items():
    print(f"Sequence: {seq}, Metrics: {metrics}")
    
# cumulative
mota = sum(metrics['MOTA'] for metrics in metrics_per_sequence.values()) / len(metrics_per_sequence)
motp = sum(metrics['MOTP'] for metrics in metrics_per_sequence.values()) / len(metrics_per_sequence)
idf1 = sum(metrics['IDF1'] for metrics in metrics_per_sequence.values()) / len(metrics_per_sequence)
print(f"Cumulative Metrics for {model_name}: MOTA: {mota:.4f}, MOTP: {motp:.4f}, IDF1: {idf1:.4f}")


run_name = params['name'] # + f"_{params['yolo_version']}{params['model_size']}" 
# Log on wandb
wandb.init(project="SharkTrack", name=run_name, config=params, job_type="training")
wandb.log({'mAP': best_mAP, 'mota': mota, 'motp': motp, 'idf1': idf1, 'track_time': track_time, 'track_device': device})
wandb.finish()