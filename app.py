from ultralytics import YOLO
import cv2

class Model():
  def __init__(self, mobile):
    """
    Args:
      mobile (bool): Whether to use lightweight model developed to run quickly on CPU
    
    Model types:
    | Type    |  Model  | Fps  |
    |---------|---------|------|
    | mobile  | Yolov8n | 2fps |
    | analyst | Yolov8s | 5fps |
    """
    mobile_model = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/yolov8_n_mvd2_50/best.pt'
    analyst_model = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/p2v5_new/weights/best.pt'

    if mobile:
      self.model_path = mobile_model
      self.tracker_path = 'botsort.yaml'
      self.device = 'cpu'
      self.fps = 2
    else:
      self.model_path = analyst_model
      self.tracker_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/trackers/custom_botsort5fps.yaml'
      self.device = '0'
      self.fps = 5
    
    # Static Hyperparameters
    self.conf_threshold = 0.2
    self.iou_association_threshold = 0.5
    self.imgsz = 640

  
  def _get_frame_skip(self, video_path):
    cap = cv2.VideoCapture(video_path)  
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / self.fps)
    return frame_skip

  
  def track(self, video_path):
    print(f'Processing video: {video_path}...')
    model = YOLO(self.model_path)

    results = model.track(
      video_path,
      conf=self.conf_threshold,
      iou=self.iou_association_threshold,
      imgsz=self.imgsz,
      tracker=self.tracker_path,
      vid_stride=self._get_frame_skip(video_path),
      device=self.self.device,
      verbose=False,
    )


  def run(self, videos_folder, stereo=False):
    processed_chapters = 0
    for video in os.listdir(base_folder):
      if os.path.isdir(video):
        for chapter in os.listdir(video):
          stereo_filter = not stereo or 'LGX' in chapter # pick only left camera
          if chapter.endswith('.mp4') and stereo_filter:
            chapter_path = os.path.join(videos_folder, video, chapter)
            self.track(chapter_path)
    
    if processed_chapters == 0:
      print('No chapters found in the given folder')
      print('Please ensure the folder structure resembles the following:')
      print('videos_folder')
      print('├── video1')
      print('│   ├── chapter1.mp4')
      print('│   ├── chapter2.mp4')
      print('└── video2')
      print('    ├── chapter1.mp4')
      print('    ├── chapter2.mp4')
    
    pass

  def get_results(self):
    # 2. From the results construct VIAME
    return self.results

def main():
  # 1. Run tracker with configs
  # 2. From the results construct VIAME

