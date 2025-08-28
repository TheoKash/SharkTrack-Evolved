from ultralytics import YOLO
import pandas as pd
import numpy as np
import time 
import os
from pathlib import Path
import sys

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from trackers.yolo import YoloTracker
from trackers.sort_adapter import Sort_adapter
from evaluation.utils import align_annotations_with_predictions_dict_corrected, target2pred_align, get_torch_device, plot_performance_graph, extract_frame_number, save_trackeval_annotations
from evaluation.TrackEval.scripts.run_mot_challenge_functional import run_mot_challenge


sequences_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/phase2'
sequences_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/frame_extraction_raw/val1/frames_5fps'
sequences_path = '/vol/biomedic3/bglocker/ugproj/tk1420/SharkTrack-Videos'
VAL_SEQUENCES = [
  'easy1',
  'easy2',
  'medium1',
  'medium2',
  'difficult1',
  'difficult2',
  'NorthSea1',
  # 'difficult3',
  # Original sequences
  # 'val1_difficult1',
  # 'val1_difficult2',
  # 'val1_easy1',
  # 'val1_easy2',
  # 'val1_medium1',
  # 'val1_medium2',
  # End of original sequences
  # 'sp_natgeo2',
  # 'gfp_hawaii1',
  # 'shlife_scalloped4',
  # 'gfp_fiji1',
  # 'shlife_smooth2',
  # 'gfp_niue1',
  # 'gfp_solomon1',
  # 'gfp_montserrat1',
  # 'gfp_rand3',
  # 'shlife_bull4'
]
tracker_class = {
  'botsort': YoloTracker,
  'bytetrack': YoloTracker,
  'sort': Sort_adapter
}



def compute_clear_metrics():
  sequence_metrics = run_mot_challenge(BENCHMARK='val1', METRICS=['CLEAR', 'Identity', 'HOTA'])
  motas, motps, idf1s, hotas = [], [], [], []
  # motas, motps, idf1s, hotas = 0, 0, 0, 0
  
  # Could aslo just return COMBINED_SEQ
  # Might need to divide by len of sequences
  
  seqToMetric = {}
  for sequence in sequence_metrics:
    if sequence in VAL_SEQUENCES:
      # exclude COMBINED_SEQ metrics      
      mota = sequence_metrics[sequence]['MOTA']
      motp = sequence_metrics[sequence]['MOTP']
      idf1 = sequence_metrics[sequence]['IDF1']
      hota = sequence_metrics[sequence]['HOTA(0)']
      
      seqToMetric[sequence] = {
        'MOTA': round(mota, 2),
        'MOTP': round(motp, 2),
        'IDF1': round(idf1, 2),
        'HOTA': round(hota, 2)
      }

  return seqToMetric

def evaluate_per_sequence(model_path, conf_threshold, iou_association_threshold, imgsz, tracker_type=None, show=False):
  all_aligned_annotations = {}
  track_time = 0
  for sequence in VAL_SEQUENCES:
    sequence_path = os.path.join(sequences_path, sequence)
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'

    annotations_path = os.path.join(sequence_path, 'annotations.csv')
    assert os.path.exists(annotations_path), f'annotations file does not exist {annotations_path}'
    annotations = pd.read_csv(annotations_path)
    
    # new ( I added the sequunce of frames from the transcoder alligned video to each file under 'frames' )
    sequence_path = os.path.join(sequence_path, 'frames')
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'
    
    print(f"Evaluating {sequence}")
    tracker_obj = tracker_class.get(tracker_type, YoloTracker) # default to YoloTracker for custom trackers
    tracker = tracker_obj(model_path, tracker_type)
    results, time = tracker.track(sequence_path, conf_threshold, iou_association_threshold, imgsz) # [bbox_xyxys, confidences, track_ids]
    track_time += time
    
    # Annotations for visualisation
    video_length =  20 if sequence != 'NorthSea1' else 19.5
    aligned_annotations = align_annotations_with_predictions_dict_corrected(annotations, results, video_length)
    all_aligned_annotations[sequence] = (aligned_annotations)
    if show:
      plot_performance_graph(aligned_annotations, sequence)

  if tracker:
    # save prediction annotations to calculate metrics
    save_trackeval_annotations(all_aligned_annotations)
    metrics_per_sequence = compute_clear_metrics()

  return metrics_per_sequence, track_time, get_torch_device(), all_aligned_annotations


def evaluate_sequence(model_path, conf_threshold, iou_association_threshold, imgsz, tracker_type):
  all_aligned_annotations = {}
  track_time = 0
  for sequence in VAL_SEQUENCES:
    sequence_path = os.path.join(sequences_path, sequence)
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'

    annotations_path = os.path.join(sequence_path, 'annotations.csv')
    assert os.path.exists(annotations_path), f'annotations file does not exist {annotations_path}'
    annotations = pd.read_csv(annotations_path)
    
    # new ( I added the sequunce of frames from the transcoder alligned video to each file under 'frames' )
    sequence_path = os.path.join(sequence_path, 'frames')
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'
    

    print(f"Evaluating {sequence}")
    tracker_obj = tracker_class.get(tracker_type, YoloTracker) # default to YoloTracker for custom trackers
    tracker = tracker_obj(model_path, tracker_type)
    results, time = tracker.track(sequence_path, conf_threshold, iou_association_threshold, imgsz) # [bbox_xyxys, confidences, track_ids]
    track_time += time
    
    # Annotations for visualisation
    video_length =  20 if sequence != 'NorthSea1' else 19.5
    aligned_annotations = align_annotations_with_predictions_dict_corrected(annotations, results, video_length)
    all_aligned_annotations[sequence] = (aligned_annotations)

    # plot_performance_graph(aligned_annotations, sequence)

  motas, motps, idf1s = 0, 0, 0 
  
  if tracker:
    # save prediction annotations to calculate metrics
    save_trackeval_annotations(all_aligned_annotations)
    metrics_per_sequence = compute_clear_metrics()
    
    motas = [metrics_per_sequence[sequence]['MOTA'] for sequence in VAL_SEQUENCES]
    motps = [metrics_per_sequence[sequence]['MOTP'] for sequence in VAL_SEQUENCES]
    idf1s = [metrics_per_sequence[sequence]['IDF1'] for sequence in VAL_SEQUENCES]
    hotas = [metrics_per_sequence[sequence]['HOTA'] for sequence in VAL_SEQUENCES]
    
  return motas, motps, idf1s, hotas, track_time, get_torch_device(), all_aligned_annotations

def evaluate(model_path, conf, iou, imgsz, tracker):
  """
  return macro-avg metrics
  """
  motas, motps, idf1s, hotas, track_time, device, _ = evaluate_sequence(model_path, conf, iou, imgsz, tracker)

  macro_mota = round(np.mean(motas), 2)
  macro_motp = round(np.mean(motps), 2)
  macro_idf1 = round(np.mean(idf1s), 2)

  return macro_mota, macro_motp, macro_idf1, track_time, device

def track(model_path, video_path, conf, iou, imgsz, tracker):
  assert tracker in ['botsort.yaml', 'bytetrack.yaml']

  model = YOLO(model_path)

  results = model.track(
    source=video_path,
    persist=True,
    conf=conf,
    iou=iou,
    imgsz=imgsz,
    tracker=str(tracker),
    verbose=False
  )

  return results

def extract_tracks(results):
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

 
