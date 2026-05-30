import argparse
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import pandas as pd

from features import build_frame_features


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


def build_one_session(
    session_id: str,
    video_path: str,
    output_path: Path,
    hand_model_path: str,
    face_model_path: str,
) -> None:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    face_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    rows = []
    prev_row = None
    frame_idx = 0

    with (
        HandLandmarker.create_from_options(hand_options) as hand_detector,
        FaceDetector.create_from_options(face_options) as face_detector,
    ):
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            h, w, _ = frame_bgr.shape

            timestamp_sec = frame_idx / fps
            timestamp_ms = int(timestamp_sec * 1000)

            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            hand_detect = hand_detector.detect_for_video(mp_image, timestamp_ms)
            face_detect = face_detector.detect_for_video(mp_image, timestamp_ms)

            row = build_frame_features(
                session_id=session_id,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                frame_width=w,
                face_detect=face_detect,
                hand_detect=hand_detect,
                prev_row=prev_row,
            )

            rows.append(row)
            prev_row = row
            frame_idx += 1

    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(rows)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="data/annotations/sessions.csv")
    parser.add_argument("--out-dir", default="data/b0_frames")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--face-model", default="models/blazeface_short_range.task")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    sessions = pd.read_csv(args.sessions)

    if args.session_id:
        sessions = sessions[sessions["session_id"] == args.session_id]

    for _, s in sessions.iterrows():
        session_id = str(s["session_id"])
        video_path = str(s["video_path"])
        output_path = Path(args.out_dir) / f"{session_id}.csv"

        build_one_session(
            session_id=session_id,
            video_path=video_path,
            output_path=output_path,
            hand_model_path=args.hand_model,
            face_model_path=args.face_model,
        )


if __name__ == "__main__":
    main()