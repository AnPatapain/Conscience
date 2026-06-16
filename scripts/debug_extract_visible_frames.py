# scripts/debug_extract_visible_frames.py

import argparse
from pathlib import Path

import cv2 as cv
import pandas as pd


def read_exact_frame(video_path: str, target_frame_idx: int):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read frame {target_frame_idx}")

        if frame_idx == target_frame_idx:
            cap.release()
            return frame

        frame_idx += 1


def draw_point(frame, x_norm, y_norm, label):
    if pd.isna(x_norm) or pd.isna(y_norm):
        return

    h, w = frame.shape[:2]
    x = int(float(x_norm) * w)
    y = int(float(y_norm) * h)

    cv.circle(frame, (x, y), 8, (0, 0, 255), -1)
    cv.putText(
        frame,
        label,
        (x + 10, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions", default="data/annotations/sessions.csv")
    parser.add_argument("--frames-dir", default="data/b0_frames")
    parser.add_argument("--out-dir", default="debug_frames")
    args = parser.parse_args()

    sessions = pd.read_csv(args.sessions)
    session = sessions[sessions["session_id"] == args.session_id]

    if session.empty:
        raise RuntimeError(f"Unknown session_id: {args.session_id}")

    video_path = str(session.iloc[0]["video_path"])
    frames_csv = Path(args.frames_dir) / f"{args.session_id}.csv"

    df = pd.read_csv(frames_csv)

    mouth_rows = df[df["mouth_visible"] == 1]
    hand_rows = df[(df["left_hand_visible"] == 1) | (df["right_hand_visible"] == 1)]

    print(mouth_rows)
    print(hand_rows)

    if mouth_rows.empty:
        print("No mouth-visible frame found")
        return

    if hand_rows.empty:
        print("No hand-visible frame found")


    first_mouth = mouth_rows.iloc[0]
    first_mouth_frame_idx = int(first_mouth["frame_idx"])
    print(
        "First mouth-visible frame:",
        first_mouth_frame_idx,
        "t=",
        float(first_mouth["timestamp_sec"]),
        "mouth_x=",
        first_mouth["mouth_x"],
        "mouth_y=",
        first_mouth["mouth_y"],
    )
    first_mouth_frame = read_exact_frame(video_path, first_mouth_frame_idx)
    draw_point(
        first_mouth_frame,
        first_mouth["mouth_x"],
        first_mouth["mouth_y"],
        "mouth",
    )

    first_hand = hand_rows.iloc[0]
    first_hand_frame_idx = int(first_hand["frame_idx"])
    print(
        "First hand-visible frame:",
        first_hand_frame_idx,
        "t=",
        float(first_hand["timestamp_sec"]),
        # left hand
        "left_wrist_x=",
        first_hand["left_wrist_x"],
        "left_wrist_y=",
        first_hand["left_wrist_y"],
        # right hand
        "right_wrist_x=",
        first_hand["right_wrist_x"],
        "right_wrist_y=",
        first_hand["right_wrist_y"],
    )
    first_hand_frame = read_exact_frame(video_path, first_hand_frame_idx)
    if first_hand["left_wrist_x"] and first_hand["left_wrist_y"]:
        draw_point(
            first_hand_frame,
            first_hand["left_wrist_x"],
            first_hand["left_wrist_y"],
            "left hand"
        )
    if first_hand["right_wrist_x"] and first_hand["right_wrist_y"]:
        draw_point(
            first_hand_frame,
            first_hand["left_wrist_x"],
            first_hand["left_wrist_y"],
            "left hand"
        )


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mouth_out_path = out_dir / f"{args.session_id}_first_mouth_frame_{first_mouth_frame_idx}.jpg"
    hand_out_path = out_dir / f"{args.session_id}_first_hand_frame_{first_hand_frame_idx}.jpg"
    cv.imwrite(str(mouth_out_path), first_mouth_frame)
    cv.imwrite(str(hand_out_path), first_hand_frame)

    print(f"Wrote {mouth_out_path}")
    print(f"Wrote {hand_out_path}")


if __name__ == "__main__":
    main()