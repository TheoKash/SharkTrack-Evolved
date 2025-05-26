from data.yolo_dataset import YoloDataset
# from interfaces import Architecture
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import numpy as np
import utils
import time
import os


# class YoloV8(Architecture):
class YoloV8():
  def __init__(self, hyperparameters, tracker):
    # super().__init__()
    self.hyperparameters = hyperparameters
    self.tracker = tracker
    
    if self.hyperparameters["pretrained"]:
      print('Loading pretrained model - no training required')
      self.model = YOLO(self.hyperparameters['model_path'])
    else:
      if self.hyperparameters["fine_tuning"]:
        # Load an existing model
        print('Loading pretrained model and fine-tuning it')
        assert self.hyperparameters["pretrained_model_path"] is not None and \
          os.path.exists(self.hyperparameters["pretrained_model_path"]) and \
          self.hyperparameters["pretrained_model_path"].endswith('.pt'), \
          'Pretrained model path must be specified and must exist'
        self.model = YOLO(self.hyperparameters["pretrained_model_path"])
      else:
        print('Training model from scratch')
        assert self.hyperparameters["model_size"] in ["n", "s", "m", "l", "x"]
        self.model = YOLO(f'yolov8{self.hyperparameters["model_size"]}.pt')

    print(f"Initialised Model {self.hyperparameters['model_name']} ")

  def __str__(self):
    return str(self.model)

  def train(self, dataset: YoloDataset):
    """ 
    Train the model using the dataset provided

    :param dataset: Dataset object containing the training data, extending torch.utils.data.Dataset
    """
    assert not self.hyperparameters['pretrained'], 'Model is already trained on this data, no need to re-training'

    # 1. Get dataset
    dataset_time = time.time()
    data_path = os.path.join(YoloDataset.experimentation_dataset_path, self.hyperparameters['training_data']['dataset_name'])
    if not self.hyperparameters['training_data']['prebuilt']:
      data_path = dataset.build()
    dataset_time = round((time.time() - dataset_time) / 60, 2)
    print(f'Dataset built in {dataset_time} minutes')

    # 2. Train on the dataset
    train_params = {
        'data': f'{data_path}/data_config.yaml',
        'epochs': self.hyperparameters['epochs'],
        'imgsz': self.hyperparameters['img_size'],
        'batch': self.hyperparameters['batch_size'],
        'patience': self.hyperparameters['patience'],
        'lr0': self.hyperparameters['lr'],
        'lrf': self.hyperparameters['lr'],
        'verbose': True,
    }

    # Take only parent folder
    print('Saving model to', self.hyperparameters['model_path'])
    model_folder = os.path.dirname(self.hyperparameters['model_path'])
    model_folder = os.path.basename(model_folder)
    train_params['project'] = model_folder
    train_params['name'] = self.hyperparameters['model_name']

    print('Starting training...')
    print('Train params:', train_params)
    start_time = time.time()
    self.model.train(**train_params)
    end_time = time.time()
    train_time = round((end_time - start_time) / 60, 2)

    # Saves model to model_path/weights/best.pt, but we want to save it to model_path/best.pt
    model_folder = self.hyperparameters['model_path']
    new_model_path = os.path.join(model_folder, 'best.pt')
    os.rename(os.path.join(self.hyperparameters['model_path'], 'weights', 'best.pt'), new_model_path)
    self.model = YOLO(os.path.join(self.hyperparameters['model_path'], 'best.pt'))
    self.hyperparameters['model_path'] = new_model_path

    # Get val mAP of the model
    # Yolo stores the epochs results in model_folder/results.csv and it stores
    # the mAP50 in the "metrics/mAP50(B)" column. 
    # The argmax("metrics/mAP50(B)") is the epoch with the highest mAP50, which
    # is the best model.
    assert model_folder.endswith(self.hyperparameters['model_name']), 'Model folder must end with model name'
    results_path = os.path.join(model_folder, 'results.csv')
    assert os.path.exists(results_path), 'Results file does not exist'
    results = pd.read_csv(results_path)
    results.columns = results.columns.str.strip()
    best_mAP = results['metrics/mAP50(B)'].max()

    return train_time, dataset_time, utils.get_torch_device(), new_model_path, best_mAP

  def evaluate(self):
    """
    1. Evaluate object detection model using the evaluation dataset
    2. Evaluate object detection model using the out-of-distribution evaluation dataset
    3. Evaluate tracker model using the evaluation dataset
    """
    bruvs_video_folder = self.bruvs_videos_folder if not self.hyperparameters['greyscale'] else self.greyscale_bruvs_videos_folder
    videos = os.listdir(bruvs_video_folder)
    assert all([video.endswith('.mp4') for video in videos])
    annotations = os.listdir(self.bruvs_annotations_folder)
    video_names = video_names = [vid[:-4] for vid in videos]
    assert len(videos) == len(annotations) and all([f'{vid}.csv' in annotations for vid in video_names])
    
    # 3. Evaluate tracker
    track_start_time = time.time()
    if self.tracker is not None:
      # macro average
      motas = []
      motps = []
      idf1s = []

      # Prepare performance plot
      # num_plots = len(video_names)
      # performance_plot, axs = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))
      model_folder = os.path.dirname(self.hyperparameters['annotations_path'])
      for i, video in enumerate(video_names):
        print(f'Evaluating {video}')
        video_path = bruvs_video_folder + video + '.mp4'
        annotations_path = self.bruvs_annotations_folder + video + '.csv'
        annotations = pd.read_csv(annotations_path)

        results = self.track(video_path)

        # Extract and store annotations for investigation
        extracted_pred_results = self._extract_tracks(results)
        aligned_annotations = utils.align_annotations_with_predictions_dict_corrected(annotations, extracted_pred_results, self.bruvs_video_length)
        aligned_annotations['frame_id'] = [i for i in range(len(aligned_annotations['gt_bbox_xyxys']))]
        aligned_annotations_df = pd.DataFrame(aligned_annotations)
        aligned_annotations_path = self.hyperparameters['annotations_path']
        aligned_annotations_df.to_csv(aligned_annotations_path, index=False)

        mota, motp, idf1, frame_avg_motp = utils.evaluate_tracking(aligned_annotations, self.hyperparameters['iou_association_threshold'])
        print(f'{video} - MOTA: {round(mota, 2)}, MOTP: {round(motp, 2)}, IDF1: {round(idf1, 2)}')
        motas.append(mota)
        motps.append(motp)
        idf1s.append(idf1)

        fig = utils.plot_performance_graph(aligned_annotations, frame_avg_motp)
        fig.savefig(os.path.join(model_folder, f"{video}.png"))

      macro_mota = round(np.mean(motas), 2)
      macro_motp = round(np.mean(motps), 2)
      macro_idf1 = round(np.mean(idf1s), 2)
      # TODO: construct average performance graph for each video! (or image of 6 combined)

    track_end_time = time.time()
    track_time = round((track_end_time - track_start_time) / 60, 2)

    return macro_mota, macro_motp, macro_idf1, track_time, utils.get_torch_device()#, performance_plot

  def track(self, video_path):
    assert self.tracker is not None
    assert str(self.tracker) in ['botsort.yaml', 'bytetrack.yaml']

    results = self.model.track(
      source=video_path,
      persist=True,
      conf=self.hyperparameters['conf_threshold'],
      iou=self.hyperparameters['iou_association_threshold'],
      imgsz=self.hyperparameters['img_size'],
      tracker=str(self.tracker),
      verbose=False
    )
    return results
    
  def _extract_tracks(self, results):
    """
    Convert Yolo-style results to list of tracks to be matched with annotation format
    :param results: List of predictions in the format results.bbox = [bbox_xyxy], [confidences], [track_ids]
    """
    bbox_xyxys = []
    confidences = []
    track_ids = []

    for i in range(len(results)):
      bbox_xyxy = []
      confidence = []
      track_id = []
      if results[i].boxes.id is not None:
        bbox_xyxy = results[i].boxes.xyxy.tolist()
        confidence = results[i].boxes.conf.tolist()
        track_id = results[i].boxes.id.tolist()

      bbox_xyxys.append(bbox_xyxy)
      confidences.append(confidence)
      track_ids.append(track_id)

    return [bbox_xyxys, confidences, track_ids]