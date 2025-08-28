from ultralytics import YOLO
import pandas as pd
import wandb
import os
import sys
from pathlib import Path

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from evaluation.evaluate_yolo_tracker import evaluate

params = {
  # 'name': 'v8s_1000e_mosaic0.8_perspective0.0005_cutmix0.1_extra_val',
  'name': 'p3v6_v8n-prod_val_iou0.5_retake_', # 'p3v6_v11_1000e_mosaic0.8_perspective0.0005_cutmix0.1',
  # 'name': 'p3v6_n2',
  'model_size': 'n', # n, s, m, l, x
  'pretrained_model': None,
  # 'pretrained_model': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/old/shark_locator_tests/runs/detect/yolov8m_mvd2/best.pt',
  # 'pretrained_model': '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8s3/weights/last.pt',
  'epochs': 1000, 
  'imgsz': 640,
  'patience': 200,
  'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/p3v6/data_config.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/p3v6_val_restored/data_config.yaml',
  # 'data_yaml': '/vol/biomedic3/bglocker/ugproj/tk1420/extra_val/data_config.yaml',
  'project_folder': 'models',
  'batch': 64,
  "iou_association_threshold": 0.5,
  "tracker": "botsort.yaml",
  "conf_threshold": 0.25,
  "yolo_version": 'yolov8',
  # "yolo_version": 'yolo11',
}



# model = YOLO(params['pretrained_model'] or f"{params['yolo_version']}{params['model_size']}.pt")  # load a pretrained model (recommended for training)
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_11n_1000e/weights/best.pt')
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_n2/weights/best.pt')
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8s3_500e/weights/best.pt')
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_11s_1000e2/weights/best.pt')
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8s_1000e2/weights/best.pt')
model = YOLO("/vol/biomedic3/bglocker/ugproj/tk1420/sharktrack/models/sharktrack.pt")
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8n_1000e_degrees15_mosaic0.75/weights/epoch250.pt')
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8n_1000e_mosaic0_perspective0.0005_cutmix0.2/weights/best.pt')
# model = YOLO("/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8n_1000e_mosaic0.8_perspective0.0005_cutmix0.15/weights/best.pt")
# model = YOLO("/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v8s_1000e_mosaic0.8_perspective0.0005_cutmix0.1/weights/best.pt")
# model = YOLO('/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Dev/models/p3v6_v11_1000e_mosaic0.8_perspective0.0005_cutmix0.1/weights/best.pt')



val_results = model.val(
  data=params['data_yaml'],
  conf=params['conf_threshold'],
  plots=True,
  name=params['name'] + '_val_conf' + str(params['conf_threshold']),
  iou=params['iou_association_threshold'],
  )

# print(val_results)
# print keys
# val_results.confusion_matrix.print()

# # Get mAP
# model_folder = os.path.join(params['project_folder'], params['name'])
# model_folder = os.path.abspath(model_folder)
# print(f'Model folder: {model_folder}')
# assert os.path.exists(model_folder), 'Model folder does not exist'

# results_path = os.path.join(model_folder, 'results.csv')
# assert os.path.exists(results_path), 'Results file does not exist'

# results = pd.read_csv(results_path)
# results.columns = results.columns.str.strip()
# best_mAP = results['metrics/mAP50(B)'].max()
# print(f"Best mAP: {best_mAP}")


# # track
# model_path = os.path.join(model_folder, 'weights', 'best.pt')
# assert os.path.exists(model_path), 'Model file does not exist'
# mota, motp, idf1, track_time, track_device = evaluate(
#   model_path, 
#   params['conf_threshold'], 
#   params['iou_association_threshold'],
#   params['imgsz'],
#   params['tracker'],
#   # params['project_folder']
# )

# print(f"mota: {mota}")
# print(f"motp: {motp}")
# print(f"idf1: {idf1}")
# print(f"track_time: {track_time}")


# run_name = params['name'] # + f"_{params['yolo_version']}{params['model_size']}" 
# # Log on wandb
# wandb.init(project="SharkTrack", name=run_name, config=params, job_type="training")
# wandb.log({'mAP': best_mAP, 'mota': mota, 'motp': motp, 'idf1': idf1, 'track_time': track_time, 'track_device': track_device})
# wandb.finish()