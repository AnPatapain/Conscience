import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


WINDOW_SIZE_SEC = 2.0
WINDOW_STRIDE_SEC = 0.5

# Rough defaults. Tune after sanity check.
WRIST_NEAR_THRESHOLD = 1.00
INDEX_NEAR_THRESHOLD = 0.55
THUMB_NEAR_THRESHOLD = 0.55
MIDDLE_NEAR_THRESHOLD = 0.55
GRIP_THRESHOLD = 0.35


def safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_min(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.min()) if len(s) else np.nan


def safe_max(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.max()) if len(s) else np.nan


def ratio(condition: pd.Series, valid: pd.Series = None) -> float:
    if valid is None:
        valid = pd.Series([True] * len(condition), index=condition.index)

    valid = valid.fillna(False)
    if int(valid.sum()) == 0:
        return 0.0

    return float((condition.fillna(False) & valid).sum() / valid.sum())


def event_float(e: pd.Series, col: str) -> float:
    v = e.get(col, np.nan)
    if pd.isna(v) or v == "":
        return np.nan
    return float(v)


def label_window(session_id: str, t: float, events: pd.DataFrame) -> Tuple[int, str, str]:
    session_events = events[events["session_id"] == session_id]

    # Prefer smoking_prep match first.
    for _, e in session_events.iterrows():
        label = str(e["label"])
        if label != "smoking_prep":
            continue

        start = event_float(e, "start_sec")
        mouth = event_float(e, "mouth_contact_sec")
        shield = event_float(e, "shield_start_sec")
        too_late = event_float(e, "too_late_sec")
        end = event_float(e, "end_sec")
        event_id = str(e["event_id"])

        if math.isnan(start) or math.isnan(too_late) or math.isnan(end):
            continue

        if start <= t <= end:
            if t <= too_late:
                if not math.isnan(mouth) and t < mouth:
                    return 1, "approach", f"smoking_prep:{event_id}"

                if not math.isnan(mouth) and not math.isnan(shield) and mouth <= t < shield:
                    return 1, "mouth_contact_pre_ignition", f"smoking_prep:{event_id}"

                if not math.isnan(shield) and shield <= t <= too_late:
                    return 1, "hand_shield_pre_ignition", f"smoking_prep:{event_id}"

                return 1, "smoking_prep_unknown_phase", f"smoking_prep:{event_id}"

            return -1, "after_ignition_ignore", f"ignore_after_too_late:{event_id}"

    # Hard negatives.
    for _, e in session_events.iterrows():
        label = str(e["label"])
        if not label.startswith("hard_negative"):
            continue

        start = event_float(e, "start_sec")
        end = event_float(e, "end_sec")
        event_id = str(e["event_id"])

        if math.isnan(start) or math.isnan(end):
            continue

        if start <= t <= end:
            return 0, "hard_negative", f"{label}:{event_id}"

    return 0, "background", "background"


def select_active_hand(w: pd.DataFrame) -> str:
    left_visible = float(w["left_hand_visible"].mean()) if "left_hand_visible" in w else 0.0
    right_visible = float(w["right_hand_visible"].mean()) if "right_hand_visible" in w else 0.0

    left_d = w["left_distance_index_mouth_norm"].dropna()
    right_d = w["right_distance_index_mouth_norm"].dropna()

    if len(left_d) == 0 and len(right_d) == 0:
        return "left" if left_visible >= right_visible else "right"

    if len(left_d) == 0:
        return "right"

    if len(right_d) == 0:
        return "left"

    return "left" if float(left_d.median()) <= float(right_d.median()) else "right"


def near_any_mouth(w: pd.DataFrame, side: str) -> pd.Series:
    wrist = w[f"{side}_distance_wrist_mouth_norm"] < WRIST_NEAR_THRESHOLD
    index = w[f"{side}_distance_index_mouth_norm"] < INDEX_NEAR_THRESHOLD
    thumb = w[f"{side}_distance_thumb_mouth_norm"] < THUMB_NEAR_THRESHOLD
    middle = w[f"{side}_distance_middle_mouth_norm"] < MIDDLE_NEAR_THRESHOLD
    return wrist | index | thumb | middle


def build_window_row(
    session_id: str,
    win_idx: int,
    start: float,
    end: float,
    w: pd.DataFrame,
    events: pd.DataFrame,
) -> Dict[str, Any]:
    active = select_active_hand(w)
    non_active = "right" if active == "left" else "left"

    active_visible = w[f"{active}_hand_visible"] == 1
    non_active_visible = w[f"{non_active}_hand_visible"] == 1
    mouth_visible = w["mouth_visible"] == 1
    visibility = active_visible & mouth_visible

    d_wrist = w[f"{active}_distance_wrist_mouth_norm"]
    d_index = w[f"{active}_distance_index_mouth_norm"]
    d_thumb = w[f"{active}_distance_thumb_mouth_norm"]
    d_middle = w[f"{active}_distance_middle_mouth_norm"]

    v_toward = w[f"{active}_velocity_wrist_toward_mouth"]
    positive_v = v_toward[v_toward > 0]

    thumb_index = w[f"{active}_thumb_index_dist_norm"]
    index_middle = w[f"{active}_index_middle_dist_norm"]

    thumb_index_close = thumb_index < GRIP_THRESHOLD
    index_middle_close = index_middle < GRIP_THRESHOLD

    active_near = near_any_mouth(w, active)
    non_active_near = near_any_mouth(w, non_active)

    # Cigarette-grip proxies: grip + either near mouth or approaching mouth.
    approaching = v_toward > 0
    cig_grip_thumb_index = thumb_index_close & (active_near | approaching)
    cig_grip_index_middle = index_middle_close & (active_near | approaching)

    y_true, phase_label, label_source = label_window(session_id, end, events)

    frame_start = int(w["frame_idx"].min()) if len(w) else -1
    frame_end = int(w["frame_idx"].max()) + 1 if len(w) else -1

    return {
        "session_id": session_id,
        "window_idx": win_idx,
        "window_start_sec": round(float(start), 3),
        "window_end_sec": round(float(end), 3),
        "frame_start": frame_start,
        "frame_end": frame_end,

        "active_hand": active,
        "active_hand_visible_ratio": float(active_visible.mean()),
        "mouth_visible_ratio": float(mouth_visible.mean()),
        "visibility_ratio": float(visibility.mean()),

        "scale_ref": safe_mean(w["scale_ref"]),

        "min_d_wrist_mouth_norm": safe_min(d_wrist),
        "mean_d_wrist_mouth_norm": safe_mean(d_wrist),
        "min_d_index_mouth_norm": safe_min(d_index),
        "mean_d_index_mouth_norm": safe_mean(d_index),

        "near_wrist_ratio": ratio(d_wrist < WRIST_NEAR_THRESHOLD, d_wrist.notna()),
        "near_index_ratio": ratio(d_index < INDEX_NEAR_THRESHOLD, d_index.notna()),

        "approach_ratio": ratio(v_toward > 0, v_toward.notna()),
        "mean_v_wrist_toward_mouth": safe_mean(v_toward),
        "peak_v_wrist_toward_mouth": safe_max(v_toward),
        "mean_positive_v_wrist_toward_mouth": safe_mean(positive_v),

        "min_d_thumb_mouth_norm": safe_min(d_thumb),
        "mean_d_thumb_mouth_norm": safe_mean(d_thumb),
        "near_thumb_ratio": ratio(d_thumb < THUMB_NEAR_THRESHOLD, d_thumb.notna()),

        "min_d_middle_mouth_norm": safe_min(d_middle),
        "mean_d_middle_mouth_norm": safe_mean(d_middle),
        "near_middle_ratio": ratio(d_middle < MIDDLE_NEAR_THRESHOLD, d_middle.notna()),

        "mean_thumb_index_dist_norm": safe_mean(thumb_index),
        "min_thumb_index_dist_norm": safe_min(thumb_index),
        "thumb_index_close_ratio": ratio(thumb_index_close, thumb_index.notna()),

        "mean_index_middle_dist_norm": safe_mean(index_middle),
        "min_index_middle_dist_norm": safe_min(index_middle),
        "index_middle_close_ratio": ratio(index_middle_close, index_middle.notna()),

        "cig_grip_thumb_index_ratio": ratio(cig_grip_thumb_index, thumb_index.notna()),
        "cig_grip_index_middle_ratio": ratio(cig_grip_index_middle, index_middle.notna()),
        "any_cig_grip_ratio": max(
            ratio(cig_grip_thumb_index, thumb_index.notna()),
            ratio(cig_grip_index_middle, index_middle.notna()),
        ),

        "non_active_hand_visible_ratio": float(non_active_visible.mean()),
        "non_active_hand_near_mouth_ratio": ratio(non_active_near, non_active_visible),
        "both_hands_near_mouth_ratio": ratio(active_near & non_active_near, active_visible | non_active_visible),

        "y_true": y_true,
        "label_source": label_source,
        "phase_label": phase_label,
    }


def build_one_session(
    session_id: str,
    frames_path: Path,
    output_path: Path,
    events: pd.DataFrame,
) -> None:
    frames = pd.read_csv(frames_path)
    if frames.empty:
        raise RuntimeError(f"Empty frames file: {frames_path}")

    max_t = float(frames["timestamp_sec"].max())

    rows = []
    win_idx = 0
    start = 0.0

    while start + WINDOW_SIZE_SEC <= max_t + 1e-9:
        end = start + WINDOW_SIZE_SEC

        w = frames[
            (frames["timestamp_sec"] >= start)
            & (frames["timestamp_sec"] < end)
        ]

        if len(w) > 0:
            rows.append(
                build_window_row(
                    session_id=session_id,
                    win_idx=win_idx,
                    start=start,
                    end=end,
                    w=w,
                    events=events,
                )
            )
            win_idx += 1

        start += WINDOW_STRIDE_SEC

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(rows)} windows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="data/annotations/sessions.csv")
    parser.add_argument("--events", default="data/annotations/events_gt.csv")
    parser.add_argument("--frames-dir", default="data/b0_frames")
    parser.add_argument("--out-dir", default="data/b0_windows")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    sessions = pd.read_csv(args.sessions)
    events = pd.read_csv(args.events)

    if args.session_id:
        sessions = sessions[sessions["session_id"] == args.session_id]

    for _, s in sessions.iterrows():
        session_id = str(s["session_id"])
        frames_path = Path(args.frames_dir) / f"{session_id}.csv"
        output_path = Path(args.out_dir) / f"{session_id}.csv"

        if not frames_path.exists():
            print(f"Skip {session_id}: missing {frames_path}")
            continue

        build_one_session(
            session_id=session_id,
            frames_path=frames_path,
            output_path=output_path,
            events=events,
        )


if __name__ == "__main__":
    main()