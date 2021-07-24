import cv2
import numpy as np

from mapper.ssc import ssc


# def extract_keypoints(frame, fast, orb, max_points=None, use_ssc=False,
#     ssc_threshold=0.1):
#     """Extracts FAST feature points and ORB descriptors in the frame."""
#     kp = fast.detect(frame, None)
#     kp = sorted(kp, key=lambda x:x.response, reverse=True)
#     if max_points is not None:
#         kp = kp[:max_points]
#     if use_ssc:
#         kp = ssc(kp, max_points, ssc_threshold,
#             frame.shape[1], frame.shape[0])
#     kp, des = orb.compute(frame, kp)
#     return kp, des


def extract_keypoints(frame, orb):
    """Extracts ORB keypoints and descriptors in the frame."""
    kp = orb.detect(frame, None)
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


# def match(bf, last_keyframe, frame, last_des, des, last_kp, kp,
#         max_apperance_distance=10.0, max_spatial_distance=10.0,
#         keep_fraction=None, draw=True):
#     matches = bf.match(last_des, des)
#     matches = sorted(matches, key=lambda x:x.distance)
#
#     print("num matches before filter ", len(matches))
#
#     # filter out matches with distance (descriptor appearance) greater than threshold
#     if max_apperance_distance is not None:
#         matches = [m for m in matches if m.distance < max_apperance_distance]
#
#     last_pts = np.array([last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
#     current_pts = np.array([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)
#
#     print("num matches after appearance filter ", len(matches))
#
#     # filter out matches with too large spatial deviation from median distance
#     if max_spatial_distance is not None:
#         spatial_dists = np.linalg.norm(last_pts - current_pts, axis=1)
#         median_dist = np.median(spatial_dists)
#         print("Median spatial distance of matches: {}".format(median_dist))
#
#         match_inlier_idxs = np.ravel(np.argwhere(
#             (spatial_dists > median_dist - max_spatial_distance) &
#             (spatial_dists < median_dist + max_spatial_distance)))
#
#         # omit filtering if more matches than specified by keep_fraction would
#         # be filtered out
#         if (keep_fraction is None) or (len(match_inlier_idxs) > keep_fraction * len(matches)):
#             matches = [matches[i] for i in match_inlier_idxs]
#             last_pts = last_pts[match_inlier_idxs]
#             current_pts = current_pts[match_inlier_idxs]
#
#     last_pts = last_pts.reshape(1, -1, 2)
#     current_pts = current_pts.reshape(1, -1, 2)
#
#     print("num matches after distance filter ", len(matches))
#     #print("Found {} matches of current frame with last key frame".format(len(matches)))
#
#     match_frame = np.zeros_like(frame)
#     if draw:
#         match_frame = cv2.drawMatches(last_keyframe, last_kp, frame, kp, matches, None)
#     return matches, last_pts, current_pts, match_frame
