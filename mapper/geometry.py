import cv2
import numpy as np


def to_twist(R, t):
    """Convert a 3x3 rotation matrix and translation vector (shape (3,))
    into a 6D twist coordinate (shape (6,))."""
    r, _ = cv2.Rodrigues(R)
    twist = np.zeros((6,))
    twist[:3] = r.reshape(3,)
    twist[3:] = t.reshape(3,)
    return twist


def from_twist(twist):
    """Convert a 6D twist coordinate (shape (6,)) into a 3x3 rotation matrix
    and translation vector (shape (3,))."""
    r = twist[:3].reshape(3, 1)
    t = twist[3:].reshape(3, 1)
    R, _ = cv2.Rodrigues(r)
    return R, t


def estimate_camera_pose(last_pts, current_pts, camera_matrix, min_inliers=20):
    """Estimate camera pose relative to last key frame by decomposing essential
    matrix computed from 2D-2D point correspondences in the current frame and
    last key frame.
    """
    # Note: tranlation t is only known up to scale
    essential_mat, mask = cv2.findEssentialMat(last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, method=cv2.LMEDS)
    num_inliers, R, t, mask = cv2.recoverPose(essential_mat, last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, mask=mask)
    mask = mask.astype(np.bool).reshape(-1,)
    print("recover pose num inliers: ", num_inliers)

    if num_inliers < min_inliers:
        raise RuntimeError("Could not recover camera pose.")
    return R, t, mask


def triangulate_map_points(last_pts, current_pts, R1, t1, R2, t2, camera_matrix):
    """Triangulate 3D map points from corresponding points in two
    keyframes. R1, t1, R2, t2 are the rotation and translation of
    the two key frames w.r.t. to the map origin.
    """
    # create projection matrices needed for triangulation of 3D points
    proj_matrix1 = np.hstack([R1.T, -R1.T.dot(t1)])
    proj_matrix2 = np.hstack([R2.T, -R2.T.dot(t2)])
    proj_matrix1 = camera_matrix.dot(proj_matrix1)
    proj_matrix2 = camera_matrix.dot(proj_matrix2)

    # triangulate new map points based on matches with previous key frame
    pts_3d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, last_pts.reshape(-1, 2).T, current_pts.reshape(-1, 2).T).T
    pts_3d = cv2.convertPointsFromHomogeneous(pts_3d).reshape(-1, 3)
    return pts_3d
