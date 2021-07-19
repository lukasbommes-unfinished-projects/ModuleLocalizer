import cv2
import numpy as np

from mapper.ssc import ssc


def extract_keypoints(frame, fast, orb, max_points=None, use_ssc=False,
    ssc_threshold=0.1):
    """Extracts FAST feature points and ORB descriptors in the frame."""
    kp = fast.detect(frame, None)
    kp = sorted(kp, key=lambda x:x.response, reverse=True)
    if max_points is not None:
        kp = kp[:max_points]
    if use_ssc:
        kp = ssc(kp, max_points, ssc_threshold,
            frame.shape[1], frame.shape[0])
    kp, des = orb.compute(frame, kp)
    return kp, des


def match(bf, last_keyframe, frame, last_des, des, last_kp, kp, distance_threshold=30.0, draw=True):
    matches = bf.match(last_des, des)
    matches = sorted(matches, key=lambda x:x.distance)
    # filter out matches with distance (descriptor appearance) greater than threshold
    matches = [m for m in matches if m.distance < distance_threshold]
    print("Found {} matches of current frame with last key frame".format(len(matches)))
    last_pts = np.array([last_kp[m.queryIdx].pt for m in matches]).reshape(1, -1, 2)
    current_pts = np.array([kp[m.trainIdx].pt for m in matches]).reshape(1, -1, 2)
    match_frame = np.zeros_like(frame)
    if draw:
        match_frame = cv2.drawMatches(last_keyframe, last_kp, frame, kp, matches, None)
    return matches, last_pts, current_pts, match_frame
