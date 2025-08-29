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
  'pretrained_model': None, # if None, use yolov8n.pt
  'epochs': 500, 
  'imgsz': 640,
  
  'patience': 150, #50
  

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
    'mosaic': 0.8, #0.8,      # mosaic
    'mixup': 0.0,             # mixup
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
}

model = YOLO(params['pretrained_model'] or f"{params['yolo_version']}{params['model_size']}.pt")  # load a pretrained model (recommended for training)        


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