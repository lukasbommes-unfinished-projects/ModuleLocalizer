import cv2
import numpy as np

from mapper.transforms import affine_matrix_from_points, decompose_matrix, \
    active_matrix_from_extrinsic_euler_xyz


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


# def invert_z_axis(pose_graph, map_points):
#     """Rotate map points and keyframe poses by 180 degree around x axis to
#     invert the z axis. Needed to align z-axes of GPS and odometry frame."""
#     nodes = list(sorted(pose_graph.nodes))
#     poses = [from_twist(pose_graph.nodes[node_id]["pose"]) for node_id in nodes]
#     R = np.array([[1, 0, 0],
#                   [0, -1, 0],
#                   [0, 0, -1]])
#     rotations = [np.matmul(R, pose[0]) for pose in poses]


def transform_to_gps_frame(pose_graph, map_points, gps):
    """Transform keyframe poses and map points into the GPS frame.

    Estimates a SIM(3) transform based on all keyframe positions and
    corresponding GPS positions.
    """
    nodes = list(sorted(pose_graph.nodes))
    poses = [from_twist(pose_graph.nodes[node_id]["pose"]) for node_id in nodes]
    positions = np.vstack([pose[1].reshape(3,) for pose in poses])
    rotations = [pose[0] for pose in poses]

    # get GPS positions of each key frame
    keyframe_idxs = [int(pose_graph.nodes[node_id]["frame_name"][6:]) for node_id in nodes]
    gps_positions = gps[np.array(keyframe_idxs), :]

    # compute scaling, rotation and translation between GPS trajectory and camera trajectory
    affine = affine_matrix_from_points(
        positions.T, gps_positions.T, shear=False, scale=True, usesvd=True)
    scale, _, angles, translate, _ = decompose_matrix(affine)
    scale = scale[0]
    R = active_matrix_from_extrinsic_euler_xyz(angles)

    # just for testing, we do not yet have enough trajectory to perform initialization
    #R = np.eye(3)
    # R = np.array([[1, 0, 0],
    #               [0, -1, 0],
    #               [0, 0, -1]])
    print("similarity transform map -> GPS frame: {}, {}, {}".format(
        scale, translate, R))

    # transform keyframe poses and map points
    map_points.pts_3d = np.matmul(R, scale*map_points.pts_3d.T).T + translate
    positions_mapped = np.matmul(R, scale*positions.T).T + translate
    rotations_mapped = [np.matmul(R, rotation) for rotation in rotations]
    for i, node_id in enumerate(nodes):
        pose_graph.nodes[node_id]["pose"] = to_twist(rotations_mapped[i], positions_mapped[i, :])
    print("pts_3d.mean: ", np.median(map_points.pts_3d, axis=0))

    return gps_positions
