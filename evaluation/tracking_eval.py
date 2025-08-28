from ultralytics import YOLO
import pandas as pd
import wandb
import os
import sys
from pathlib import Path
from evaluate_yolo_tracker import evaluate_per_sequence
import wandb

model_names = ['p3v6_v8n_1000e_augs_ns_backgrounds',
               'p3v6_v8s_1000e_v8s_augs_ns_backgrounds8',
               'p3v6_v8s_1000e_mosaic0.8_perspective0.0005_cutmix0.1',
               'p3v6_v8n_1000e_mosaic0.8_perspective0.0005_cutmix0.15',
               'p3v6_v8s_1000e2',
               'p3v6_v11_1000e_mosaic0.8_perspective0.0005_cutmix0.1',
               ]
modelToConf = {'p3v6_v8n_1000e_augs_ns_backgrounds': 0.11328125, 
               'p3v6_v8s_1000e_v8s_augs_ns_backgrounds8': 0.06640625, 
               'p3v6_v8s_1000e_mosaic0.8_perspective0.0005_cutmix0.1': 0.1015625, 
               'p3v6_v8n_1000e_mosaic0.8_perspective0.0005_cutmix0.15': 0.10546875, 
               'p3v6_v8s_1000e2': 0.005859375, 
               'p3v6_v11_1000e_mosaic0.8_perspective0.0005_cutmix0.1': 0.09765625}


def load_model(model_version):
    project = '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/'
    model_folder_path = os.path.join(project, 'models', model_version)
    model_path = os.path.join(model_folder_path, 'weights/best.pt')
    return YOLO(model_path)


# model_folder_path = os.path.join('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models', model_name)
# yaml_path = os.path.join(model_folder_path, 'args.yaml')

# inspect the model args.yaml to get the following params
# with open(yaml_path, 'r') as f:
#     args = f.read()
# args = dict(line.strip().split(': ') for line in args.splitlines() if ': ' in line)
# model_size = args.get('model', 'a_name_n.pt').split('.')[0][-1]

# modelToParams = {}
# modelToParams[model_name] = {
#     'name': model_name,
#     'model_size': model_size,
#     'pretrained_model': args.get('model', 'null'),
#     'epochs': args.get('epochs', 1000),
#     'imgsz': 640,
#     'patience': args.get('patience', 50),
#     'data_yaml': args.get('data', '/vol/biomedic3/bglocker/ugproj/tk1420/p3v6/data_config.yaml'),
#     'project_folder': 'models',
#     'batch': args.get('batch', 64),
#     'iou_association_threshold': 0.5,
#     'tracker': 'botsort.yaml',
#     'conf_threshold': conf,
#     'yolo_version': 'yolov8',
#     'augmentations': {
#         'hsv_h': args.get('hsv_h', 0.015),
#         'hsv_s': args.get('hsv_s', 0.7),
#         'hsv_v': args.get('hsv_v', 0.4),
#         'degrees': args.get('degrees', 0.0),
#         'translate': args.get('translate', 0.1),
#         'scale': args.get('scale', 0.5),
#         'shear': args.get('shear', 0.0),
#         'perspective': args.get('perspective', 0.0),
#         'flipud': args.get('flipud', 0.0),
#         'fliplr': args.get('fliplr', 0.5),
#         'bgr': args.get('bgr', 0.0),
#         'mosaic': args.get('mosaic', 1.0),
#         'mixup': args.get('mixup', 0.0),
#         'cutmix': args.get('cutmix', 0.0),
#     }
# }

conf = 0.1
# conf = 0.06

data_yaml = '/vol/biomedic3/bglocker/ugproj/tk1420/ALv3.yaml'
data_yaml = '/vol/biomedic3/bglocker/ugproj/tk1420/ALv4.yaml'

# model_name = 'ALv3_v1-v8n_augs_300e'
# model_name = 'ALv4_v8n_augs'
# model_name = 'harsh_lighting_test20'
model_name = 'p3v6_n2'

project_folder = '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/'

print(f"Evaluating model {model_name} at confidence threshold {conf:.4f}")
model = load_model(model_name)
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/sharktrack/models/sharktrack.pt')
# model_path = '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/ALv3_v1-v8n_augs10/weights/epoch300.pt'
# model = YOLO(model_path)

job_name='local_eval'

# Evaluate the model on the SharkTrack dataset
results = model.val(name=f"{job_name}_{model_name}_val_conf{conf:.4f}_",
                    data=data_yaml,
                    conf=conf,
                    plots=True,
                    save_json=False,
                    verbose=False,
                    iou=0.5)

# Print confusion matrix
cm = results.confusion_matrix.matrix
print(f"Confusion Matrix for {model_name} at conf: {conf:.4f}")
print(cm)
print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']}")
print("Exiting early pre tracking.")
# exit()

# Evaluate per sequence metrics
# model_path = os.path.join(project_folder, 'models', model_name, 'weights', 'best.pt')
# model_path = "/vol/biomedic3/bglocker/ugproj/tk1420/sharktrack/models/sharktrack.pt"
model_path = os.path.join(project_folder, 'models', model_name, 'weights', 'best.pt')

metrics_per_sequence, track_time, device, all_aligned_annotations = evaluate_per_sequence(model_path, conf, iou_association_threshold=0.5,imgsz=640, tracker_type='tracker_5fps.yaml')
print(f"Per-sequence metrics for {model_name}:")
for seq, metrics in metrics_per_sequence.items():
    print(f"Sequence: {seq}, Metrics: {metrics}")
    
# cumulative
mota = sum(metrics['MOTA'] for metrics in metrics_per_sequence.values()) / len(metrics_per_sequence)
motp = sum(metrics['MOTP'] for metrics in metrics_per_sequence.values()) / len(metrics_per_sequence)
idf1 = sum(metrics['IDF1'] for metrics in metrics_per_sequence.values()) / len(metrics_per_sequence)
print(f"Cumulative Metrics for {model_name}: MOTA: {mota:.4f}, MOTP: {motp:.4f}, IDF1: {idf1:.4f}")
    
# Log to wandb
# run_name = model_name + f"_conf{conf:.4f}_{job_name}"
# wandb.init(project="SharkTrack-Dev", name=run_name, config=modelToParams[model_name], job_type=job_name)
# log_data = {}
# for seq, metrics in metrics_per_sequence.items():
#     # Best map50
#     log_data[f"{seq}/map50"] = results.results_dict['metrics/mAP50(B)']
#     log_data[f"{seq}/mota"] = metrics['MOTA']
#     log_data[f"{seq}/motp"] = metrics['MOTP']
#     log_data[f"{seq}/idf1"] = metrics['IDF1']
    
# log_data['track_time'] = track_time
# log_data['track_device'] = device
# wandb.log(log_data)
# wandb.finish()
# print(f"Logged results for {model_name} to wandb.")