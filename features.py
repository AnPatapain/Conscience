import math
from typing import Any, Dict, Optional, Tuple


MOUTH_KEYPOINT_INDEX = 3

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12


Point = Tuple[float, float]


def nan() -> float:
    return float("nan")


def is_valid_point(p: Optional[Point]) -> bool:
    return p is not None and not any(math.isnan(v) for v in p)


def point_xy(landmark: Any) -> Point:
    return float(landmark.x), float(landmark.y)


def dist(p1: Optional[Point], p2: Optional[Point]) -> float:
    if not is_valid_point(p1) or not is_valid_point(p2):
        return nan()
    return float(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def norm_dist(p1: Optional[Point], p2: Optional[Point], scale_ref: float) -> float:
    d = dist(p1, p2)
    if math.isnan(d) or scale_ref <= 1e-6 or math.isnan(scale_ref):
        return nan()
    return d / scale_ref


def get_face_mouth_and_scale(face_detect: Any, frame_width: int) -> Tuple[Optional[Point], float]:
    """
    Uses MediaPipe FaceDetector output.
    Mouth = keypoint 3.
    scale_ref = face width proxy.
    Prefer tragion distance if available, fallback to bbox width.
    """
    if face_detect is None or not face_detect.detections:
        return None, nan()

    det = face_detect.detections[0]

    mouth = None
    if det.keypoints and len(det.keypoints) > MOUTH_KEYPOINT_INDEX:
        mouth = point_xy(det.keypoints[MOUTH_KEYPOINT_INDEX])

    # Face keypoints:
    # 0 right eye, 1 left eye, 2 nose, 3 mouth, 4 left tragion, 5 right tragion
    scale_ref = nan()
    if det.keypoints and len(det.keypoints) > 5:
        left_tragion = point_xy(det.keypoints[4])
        right_tragion = point_xy(det.keypoints[5])
        scale_ref = dist(left_tragion, right_tragion)

    # Fallback: bounding box width normalized by frame width
    if math.isnan(scale_ref) or scale_ref <= 1e-6:
        bbox = getattr(det, "bounding_box", None)
        if bbox is not None and frame_width > 0:
            scale_ref = float(bbox.width) / float(frame_width)

    return mouth, scale_ref


def handedness_name(handedness: Any) -> Optional[str]:
    """
    Robust enough for MediaPipe Tasks handedness format.
    """
    if not handedness:
        return None

    cat = handedness[0]
    name = getattr(cat, "category_name", None) or getattr(cat, "display_name", None)
    if not name:
        return None

    return str(name).lower()


def split_hands(hand_detect: Any) -> Dict[str, Optional[Any]]:
    """
    Return {"left": landmarks_or_None, "right": landmarks_or_None}.

    If handedness is unavailable, fallback by wrist x-position:
    smaller x = left side of image, larger x = right side of image.
    """
    result = {"left": None, "right": None}

    if hand_detect is None or not hand_detect.hand_landmarks:
        return result

    hands = hand_detect.hand_landmarks
    handedness_list = getattr(hand_detect, "handedness", None)

    unresolved = []

    for i, landmarks in enumerate(hands):
        hname = None
        if handedness_list and i < len(handedness_list):
            hname = handedness_name(handedness_list[i])

        if hname in ("left", "right") and result[hname] is None:
            result[hname] = landmarks
        else:
            unresolved.append(landmarks)

    # Fallback assignment by wrist x if needed
    for landmarks in unresolved:
        wrist_x = point_xy(landmarks[WRIST])[0]
        side = "left" if wrist_x < 0.5 else "right"
        if result[side] is None:
            result[side] = landmarks
        else:
            other = "right" if side == "left" else "left"
            if result[other] is None:
                result[other] = landmarks

    return result


def hand_points(landmarks: Optional[Any]) -> Dict[str, Optional[Point]]:
    if landmarks is None:
        return {
            "wrist": None,
            "thumb": None,
            "index": None,
            "middle": None,
        }

    return {
        "wrist": point_xy(landmarks[WRIST]),
        "thumb": point_xy(landmarks[THUMB_TIP]),
        "index": point_xy(landmarks[INDEX_TIP]),
        "middle": point_xy(landmarks[MIDDLE_TIP]),
    }


def empty_hand_features(prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_hand_visible": 0,

        f"{prefix}_wrist_x": nan(),
        f"{prefix}_wrist_y": nan(),

        f"{prefix}_distance_wrist_mouth_norm": nan(),
        f"{prefix}_distance_index_mouth_norm": nan(),
        f"{prefix}_distance_middle_mouth_norm": nan(),
        f"{prefix}_distance_thumb_mouth_norm": nan(),

        f"{prefix}_thumb_index_dist_norm": nan(),
        f"{prefix}_index_middle_dist_norm": nan(),

        f"{prefix}_velocity_wrist_toward_mouth": nan(),
        f"{prefix}_speed_wrist_norm": nan(),
        f"delta_{prefix}_wrist_mouth_norm": nan(),
    }


def build_frame_features(
    session_id: str,
    frame_idx: int,
    timestamp_sec: float,
    frame_width: int,
    face_detect: Any,
    hand_detect: Any,
    prev_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    mouth, scale_ref = get_face_mouth_and_scale(face_detect, frame_width)
    mouth_visible = int(is_valid_point(mouth))

    row: Dict[str, Any] = {
        "session_id": session_id,
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,

        "mouth_visible": mouth_visible,
        "left_hand_visible": 0,
        "right_hand_visible": 0,

        "mouth_x": mouth[0] if mouth else nan(),
        "mouth_y": mouth[1] if mouth else nan(),
        "scale_ref": scale_ref,
    }

    hands = split_hands(hand_detect)

    for side in ("left", "right"):
        row.update(empty_hand_features(side))

        pts = hand_points(hands[side])
        if pts["wrist"] is None:
            continue

        row[f"{side}_hand_visible"] = 1
        row[f"{side}_wrist_x"] = pts["wrist"][0]
        row[f"{side}_wrist_y"] = pts["wrist"][1]

        row[f"{side}_distance_wrist_mouth_norm"] = norm_dist(pts["wrist"], mouth, scale_ref)
        row[f"{side}_distance_index_mouth_norm"] = norm_dist(pts["index"], mouth, scale_ref)
        row[f"{side}_distance_middle_mouth_norm"] = norm_dist(pts["middle"], mouth, scale_ref)
        row[f"{side}_distance_thumb_mouth_norm"] = norm_dist(pts["thumb"], mouth, scale_ref)

        row[f"{side}_thumb_index_dist_norm"] = norm_dist(pts["thumb"], pts["index"], scale_ref)
        row[f"{side}_index_middle_dist_norm"] = norm_dist(pts["index"], pts["middle"], scale_ref)

        if prev_row is None:
            continue

        prev_t = float(prev_row["timestamp_sec"])
        dt = max(timestamp_sec - prev_t, 1e-6)

        prev_wx = prev_row.get(f"{side}_wrist_x", nan())
        prev_wy = prev_row.get(f"{side}_wrist_y", nan())

        if not math.isnan(prev_wx) and not math.isnan(prev_wy) and is_valid_point(mouth):
            wx, wy = pts["wrist"]

            vx = (wx - prev_wx) / dt
            vy = (wy - prev_wy) / dt

            # Normalize velocity by face scale
            if not math.isnan(scale_ref) and scale_ref > 1e-6:
                vx /= scale_ref
                vy /= scale_ref

            ux = mouth[0] - wx
            uy = mouth[1] - wy
            u_norm = math.sqrt(ux * ux + uy * uy)

            if u_norm > 1e-6:
                ux /= u_norm
                uy /= u_norm
                row[f"{side}_velocity_wrist_toward_mouth"] = vx * ux + vy * uy
                row[f"{side}_speed_wrist_norm"] = math.sqrt(vx * vx + vy * vy)

        prev_d = prev_row.get(f"{side}_distance_wrist_mouth_norm", nan())
        curr_d = row.get(f"{side}_distance_wrist_mouth_norm", nan())
        if not math.isnan(prev_d) and not math.isnan(curr_d):
            row[f"delta_{side}_wrist_mouth_norm"] = curr_d - prev_d

    return row