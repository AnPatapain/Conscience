import json
import numpy as np
import time
import cv2 as cv
import mediapipe as mp
from types import SimpleNamespace
from collections import deque

# Base options and config
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Config for Hand landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hlOptions = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)


# Config for Face detection
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
fdOptions = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='models/blazeface_short_range.task'),
    running_mode=VisionRunningMode.VIDEO)

# Config
MAX_TEMPORAL_WINDOW_SIZE=16
HAND_NEAR_MOUTH_THRESHOLD=0.20
PERSISTENCE_THRESHOLD=0.5
PERSISTENCE_PENALTY=0.5

SMOKING_HAND_LANDMARK_INDEX = [
  7,   # INDEX_FINGER_DIP
  8,   # INDEX_FINGER_TIP
  11,  # MIDDLE_FINGER_DIP
  12,  # MIDDLE_FINGER_TIP
]
MOUTH_KEYPOINT_INDEX = 3


def safe_point_xy(p):
  return p.x, p.y


def get_smoking_fingers(hand_landmarks):
  if not hand_landmarks: return []
  return [ safe_point_xy(hand_landmarks[i]) for i in SMOKING_HAND_LANDMARK_INDEX ]

def euclidean_distance(p1, p2):
  return ((p1[0] - p2[0])**2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_first_hand_landmarks(hand_detect):
  """
    MediaPipe Tasks HandLandmarkerResult:
    hand_detect.hand_landmarks = [
        [landmark0, landmark1, ..., landmark20],
        ...
    ]
    """
  if hand_detect is None: return None
  if not hand_detect.hand_landmarks: return None
  return hand_detect.hand_landmarks[0]

def get_first_mouth_point(face_detect):
  """
    MediaPipe Tasks FaceDetectorResult:
    face_detect.detections = [detection0, detection1, ...]
    detection.keypoints usually contains:
    0 right eye
    1 left eye
    2 nose tip
    3 mouth center
    4 left eye tragion
    5 right eye tragion
  """
  if not face_detect: return None
  if not face_detect.detections: return None
  
  detection = face_detect.detections[0]
  
  if not detection.keypoints or len(detection.keypoints) <= MOUTH_KEYPOINT_INDEX:
    return None
  
  return safe_point_xy(detection.keypoints[MOUTH_KEYPOINT_INDEX])

def build_feat_vec(
  hand_detect,
  face_detect,
):
  '''
  Build feature vector for each frame.
  '''
  hand_landmarks = get_first_hand_landmarks(hand_detect)
  mouth_keypoint = get_first_mouth_point(face_detect)
  
  hand_found = hand_landmarks is not None
  face_found = mouth_keypoint is not None
  
  finger_points = get_smoking_fingers(hand_landmarks)
  
  centroid2mouth = None
  minFinger2Mouth = None
  
  if hand_found and face_found and len(finger_points) > 0:
    centroid = (
      sum(p[0] for p in finger_points)/len(finger_points),
      sum(p[1] for p in finger_points)/len(finger_points)
    )
    centroid2mouth = euclidean_distance(centroid, mouth_keypoint)
    minFinger2Mouth = min(
      euclidean_distance(p, mouth_keypoint) for p in finger_points
    )
  
  return {
    "face": face_found,
    "hand": hand_found,
    "cigarette": False,
    "cigarette_in_hand": False,
    "centroid_to_mouth": centroid2mouth,
    "min_finger_to_mouth": minFinger2Mouth,
  }
  
def heuristic_model(temporal_window):
  near_count = 0
  valid_count = 0
  for feat_vec in temporal_window:
    if (
      not feat_vec["face"]
      or not feat_vec["hand"]
      or not feat_vec["centroid_to_mouth"]
      or not feat_vec["min_finger_to_mouth"]
    ): continue
   
    valid_count += 1
    
    hand_near_mouth = (feat_vec["centroid_to_mouth"] < HAND_NEAR_MOUTH_THRESHOLD or
                       feat_vec["min_finger_to_mouth"] < HAND_NEAR_MOUTH_THRESHOLD)
    if hand_near_mouth:
      near_count += 1
  
  # Early return to avoid division by 0 
  if valid_count == 0: return 0.0, 0.0
  
  persistence = near_count / valid_count
  if persistence >= PERSISTENCE_THRESHOLD:
    score = persistence
  else:
    score = persistence * PERSISTENCE_PENALTY
  return score, persistence


def print_window_table(window):
    if not window:
        print("<empty window>")
        return

    fields = list(window[0].keys())

    def fmt(value):
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    print("idx | " + " | ".join(fields))
    print("-" * (6 + sum(len(f) + 3 for f in fields)))

    for i, feat_vec in enumerate(window):
        values = [fmt(feat_vec.get(field)) for field in fields]
        print(f"{i:>3} | " + " | ".join(values))

# ---------------------
# Main live loop
# ---------------------
def main():
  cap = cv.VideoCapture(0)
  temporal_window = deque(maxlen=MAX_TEMPORAL_WINDOW_SIZE)
  
  with HandLandmarker.create_from_options(hlOptions) as handDetector, FaceDetector.create_from_options(fdOptions) as faceDetector:
    while cap.isOpened():
      success, frame_bgr = cap.read()

      if not success:
        print("Cannot receive frame. Exit")
        break
      
      # prepare data for detectors
      frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
      
      # run detectors
      timestamp_ms = int(time.time() * 1000)
      hand_detect = handDetector.detect_for_video(mp_image, timestamp_ms)
      face_detect = faceDetector.detect_for_video(mp_image, timestamp_ms)
      
      # Build feature vector
      feat_vec = build_feat_vec(hand_detect, face_detect)
      temporal_window.append(feat_vec)
      if len(temporal_window) < MAX_TEMPORAL_WINDOW_SIZE:
        continue
      
      # Input of behavioral model
      X_t = list(temporal_window)
      print_window_table(X_t)
      
      # Feed input to behavioral model - Heuristic currently
      raw_score, persistence = heuristic_model(X_t)
      
      observe_packet = {
        "observe_confidence_raw": raw_score,
        "temporal_persistence": persistence,
      }  
      print(json.dumps(observe_packet))
      

      # visualize
      height, width, _ = frame_bgr.shape
      for hand_landmarks in hand_detect.hand_landmarks:
        for landmark in hand_landmarks:
          x = int(landmark.x * width)
          y = int(landmark.y * height)
          cv.circle(frame_bgr, (x, y), 4, (0, 255, 0))

      for detection in face_detect.detections:
        for landmark in detection.keypoints:
          x = int(landmark.x * width)
          y = int(landmark.y * height)
          cv.circle(frame_bgr, (x, y), 4, (255, 0, 0))


      cv.imshow("hand landmarks", frame_bgr)

      # On ESC key
      if cv.waitKey(1) == 27:
        break

  cap.release()
  cv.destroyAllWindows()

if __name__ == "__main__":
  main()