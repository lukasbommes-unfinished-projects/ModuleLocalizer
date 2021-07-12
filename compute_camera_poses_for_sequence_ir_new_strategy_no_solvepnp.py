import os
import glob
import pickle
import numpy as np
import cv2
import networkx as nx

from ssc import ssc
from common import Capture

camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
dist_coeffs = pickle.load(open("camera_calibration/parameters/ir/dist_coeffs.pkl", "rb"))

frames_root = "/storage/data/splitted/20210510_Schmalenbach/02_north"
frame_files = sorted(glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
cap = Capture(frame_files, None, camera_matrix, dist_coeffs)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
num_ret_points = 3000
tolerance = 0.1
match_max_distance = 30.0


def get_frame(cap):
    """Reads and undistorts next frame from stream."""
    frame, _, _, _ = cap.get_next_frame(
        preprocess=True, undistort=True, equalize_hist=True)
    return frame


def extract_kp_des(frame, fast, orb):
    kp = fast.detect(frame, None)
    kp = sorted(kp, key = lambda x:x.response, reverse=True)
    #kp = ssc(kp, num_ret_points, tolerance, frame.shape[1], frame.shape[0])
    kp, des = orb.compute(frame, kp)
    return kp, des


def match(bf, last_keyframe, current_frame, last_des, des, last_kp, kp, distance_threshold=30.0, draw=True):
    matches = bf.match(last_des, des)
    matches = sorted(matches, key = lambda x:x.distance)
    # filter out matches with distance (descriptor appearance) greater than threshold
    matches = [m for m in matches if m.distance < distance_threshold]
    print("Found {} matches of current frame with last key frame".format(len(matches)))
    last_pts = np.array([last_kp[m.queryIdx].pt for m in matches]).reshape(1, -1, 2)
    current_pts = np.array([kp[m.trainIdx].pt for m in matches]).reshape(1, -1, 2)
    match_frame = np.zeros_like(current_frame)
    if draw:
        match_frame = cv2.drawMatches(last_keyframe, last_kp, current_frame, kp, matches, None)
    return matches, last_pts, current_pts, match_frame


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


def dump_result(path="."):
    pickle.dump(Rs, open(os.path.join(path, "Rs.pkl"), "wb"))
    pickle.dump(ts, open(os.path.join(path, "ts.pkl"), "wb"))
    pickle.dump(map_points, open(os.path.join(path, "map_points.pkl"), "wb"))

    # extract keyframe poses and visible map points from pose graph for plotting
    kf_poses = [data["pose"] for _, data in pose_graph.nodes.data()]
    pickle.dump(kf_poses, open(os.path.join(path, "kf_poses.pkl"), "wb"))
    #kf_visible_map_points = [data["visible_map_points"] for _, data in pose_graph.nodes.data()]
    #pickle.dump(kf_visible_map_points, open(os.path.join(path, "kf_visible_map_points.pkl"), "wb"))
    kf_frames = [data["frame"] for _, data in pose_graph.nodes.data()]
    pickle.dump(kf_frames, open(os.path.join(path, "kf_frames.pkl"), "wb"))
    #kf_kp_matched = [data["kp_matched"] for _, data in pose_graph.nodes.data()]
    #pickle.dump(kf_kp_matched, open(os.path.join(path, "kf_kp_matched.pkl"), "wb"))



class MapPoints:
    """Map points

    Attributes:

        idx (`numpy.ndarray`): Shape (-1,). Indices of map points ranging from
            0 to N-1 for N map points.

        pts_3d (`numpy.ndarray`): Shape (-1, 3). The (X, Y, Z) coordinates of
            each map point in the world reference frame.

        observing_keyframes (`list` of `list` of `int`): Contains one sublist
            for each map point. Each sublist stores the indices of the key frames
            which observe the map point. These key frame indices correspond to
            the node index of the key frame in the pose graph.

        associated_kp_indices (`list` of `list` of `int`): Contains one sublist
            for each map point. Each sublist stores the indices of the key points
            within the observing keyframes from which the map point was triangulated.
            E.g. given the observing key frames [0, 1, 2] for a map point, the
            associated_kp_indices sublist [113, 20, 5] means that the map point
            corresponds to key point 113 in key frame 0, key point 20 in key frame 1
            and key point 5 in key frame 2.
    """
    def __init__(self):
        """Map points"""
        self.idx = None
        self.pts_3d = np.empty(shape=(0, 3), dtype=np.float64)
        self.observing_keyframes = []
        self.associated_kp_indices = []


    def insert(self, new_map_points, associated_kp_indices, observing_kfs):
        """Add new map points into the exiting map."""
        if self.idx is not None:
            self.idx = np.hstack((self.idx, np.arange(self.idx[-1]+1, self.idx[-1]+1+new_map_points.shape[0])))
        else:
            self.idx = np.arange(0, new_map_points.shape[0])
        self.pts_3d = np.vstack((self.pts_3d, new_map_points))
        for _ in range(new_map_points.shape[0]):
            self.observing_keyframes.append(observing_kfs)
        self.associated_kp_indices.extend(associated_kp_indices)


    def get_by_observation(self, keyframe_idx):
        """Get all map points observed by a keyframe."""
        # get indices of all map points which are observed by the query keyframe
        result_idx = np.array([i for i, mp in enumerate(self.observing_keyframes) if keyframe_idx in mp])
        idx = self.idx[result_idx]
        pts_3d = self.pts_3d[result_idx, :]
        # get indices of keypoints associated to the map point in the query keyframe
        pos_idx = [mp.index(keyframe_idx) for mp in self.observing_keyframes if keyframe_idx in mp]
        associated_kp_indices = [self.associated_kp_indices[r][p] for r, p in zip(result_idx, pos_idx)]
        return idx, pts_3d, associated_kp_indices


cv2.namedWindow("last_keyframe", cv2.WINDOW_NORMAL)
cv2.resizeWindow("last_keyframe", 1600, 900)
cv2.namedWindow("current_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("current_frame", 1600, 900)
cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("match_frame", 1600, 900)

step_wise = True
match_frame = None

Rs = []
ts = []


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


# TODO: Since a planar scene is observed, using the essential matrix is not optimal
# higher accuracy can be achieved by estimating a homography (cv2.findHomography)
# and decomposing the homography into possible rotations and translations (see experiments/Untitled9.ipynb)
def initialize(fast, orb, camera_matrix, min_parallax=60.0):
    """Initialize two keyframes, the camera poses and a 3D point cloud.

    Args:
        min_parallax (`float`): Threshold for the median distance of all
            keypoint matches between the first keyframe (firs frame) and the
            second key frame. Is used to determine which frame is the second
            keyframe. This is needed to ensure enough parallax to recover
            the camera poses and 3D points.
    """
    pose_graph = nx.Graph()  # stores keyframe poses and data (keypoints, ORB descriptors, etc.)
    map_points = MapPoints()  # stores 3D world points

    # get first key frame
    frame = get_frame(cap)
    kp, des = extract_kp_des(frame, fast, orb)
    pose_graph.add_node(0, frame=frame, kp=kp, des=des)

    frame_idx_init = 0

    while True:
        frame = get_frame(cap)
        if frame is None:
            break
        frame_idx_init += 1

        # extract keypoints and match with first key frame
        kp, des = extract_kp_des(frame, fast, orb)
        matches, last_pts, current_pts, match_frame = match(bf,
            pose_graph.nodes[0]["frame"], frame, pose_graph.nodes[0]["des"],
            des, pose_graph.nodes[0]["kp"], kp, match_max_distance, draw=False)

        # determine median distance between all matched feature points
        median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
        print(median_dist)

        # if distance exceeds threshold choose frame as second keyframe
        if median_dist >= min_parallax:
            pose_graph.add_node(1, frame=frame, kp=kp, des=des)
            break

    pose_graph.add_edge(0, 1, num_matches=len(matches))

    # separately store the keypoints in matched order for tracking later
    #pose_graph.nodes[0]["kp_matched"] = last_pts.reshape(-1, 2)
    #pose_graph.nodes[1]["kp_matched"] = current_pts.reshape(-1, 2)

    # compute relative camera pose for second frame
    R, t, mask = estimate_camera_pose(last_pts, current_pts, camera_matrix, min_inliers=20)

    # filter inliers
    last_pts = last_pts[:, mask, :]
    current_pts = current_pts[:, mask, :]
    matches = list(np.array(matches)[mask])

    # relative camera pose
    R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float64)
    t1 = np.array([[0], [0], [0]]).astype(np.float64)
    R2 = R.T
    t2 = -np.matmul(R.T, t.reshape(3,)).reshape(3,1)

    # insert pose (in twist coordinates) of KF0 and KF1 into keyframes dict
    # poses are w.r.t. KF0 which is the base coordinate system of the entire map
    pose_graph.nodes[0]["pose"] = to_twist(R1, t1)
    pose_graph.nodes[1]["pose"] = to_twist(R2, t2)

    # triangulate initial 3D point cloud
    pts_3d = triangulate_map_points(last_pts, current_pts, R1, t1, R2, t2, camera_matrix)

    # filter outliers
    #pts_3d = remove_outliers(pts_3d)

    # add triangulated points to map points
    associated_kp_indices = [[m.queryIdx, m.trainIdx] for m in matches]
    map_points.insert(pts_3d, associated_kp_indices, observing_kfs=[0, 1])  # triangulatedmap points between KF0 and KF1

    # Store indices of map points belonging to KF0 and KF1 in pose graph node
    #pose_graph.nodes[0]["visible_map_points"] = [range(0, pts_3d.shape[0])]
    #pose_graph.nodes[1]["visible_map_points"] = [range(0, pts_3d.shape[0])]

    print("Initialization successful. Chose frames 0 and {} as key frames".format(frame_idx_init))

    # TODO: perform full BA to optimize initial camera poses and map points [see. ORB_SLAM IV. 5)]

    return pose_graph, map_points, frame_idx_init


def get_map_points_and_kps_for_matches(last_kf_index, matches):
    """Returns map points and corresponding key points for current frame.

    Given matches between a current frame and the last keyframe the function
    finds which key point in the current frame correpsonds to which key point
    in the last key frame and returns the map points correpsodning to these
    key points. It also returns the indices in the `matches` array corresponding
    to the returned 3D points.
    """
    # get all map points observed in last KF
    _, pts_3d, associated_kp_indices = map_points.get_by_observation(last_kf_index)  # get all map points observed by last KF
    # get indices of map points which were found again in the current frame
    kp_idxs = []
    new_matches = []
    match_idxs = []
    for match_idx, match in enumerate(matches):
        try:
            kp_idx = associated_kp_indices.index(match.queryIdx)
        except ValueError:
            pass
        else:
            kp_idxs.append(kp_idx)
            match_idxs.append(match_idx)
            new_matches.append(match)
    print("{} of {} ({:3.3f} %) keypoints in last key frame have been found again in current frame".format(len(new_matches), len(matches), len(new_matches)/len(matches)))
    # get map points according to the indices
    pts_3d = pts_3d[np.array(kp_idxs), :]
    # get corresponding key points in the current frame
    #img_points = np.array([current_kp[m.trainIdx].pt for m in new_matches]).reshape(-1, 1, 2)
    return pts_3d, np.array(match_idxs)


################################################################################

pose_graph, map_points, frame_idx = initialize(fast, orb, camera_matrix)
current_kp = None

while(True):

    try:

        # TODO: whenever a new frame arrives
        # 1) preprocess frame (warp)
        # 2) extract kps, track kps
        # 3) match kps with kps of previous KF
        # 4) solve PnP for matched keypoints to recover camera pose of current frame
        # 5) compute the relative change of pose w.r.t pose in last KF
        # 6) if pose change exceeds threshold insert a new keyframe

        # TODO: insert a new key frame
        # 1) triangulate a new set of 3D points between the last KF and this KF
        # 2) merge the 3D point cloud with existing map points (3D points), remove duplicates, etc.
        # 3) check which other keyframes share the same keypoints (pyDBoW3)
        # 4) perform BA with local group (new keyframe + keyframes from 1) to adjust pose and 3D point estimates of the local group

        current_frame = get_frame(cap)
        if current_frame is None:
            break

        frame_idx += 1

        print("frame", frame_idx)

### local tracking (initial pose estimate)

        # get initial pose estimate by matching keypoints with previous KF
        current_kp, current_des = extract_kp_des(current_frame, fast, orb)
        prev_node_id = sorted(pose_graph.nodes)[-1]
        matches, last_pts, current_pts, match_frame = match(bf,
            pose_graph.nodes[prev_node_id]["frame"],
            current_frame,
            pose_graph.nodes[prev_node_id]["des"],
            current_des, pose_graph.nodes[prev_node_id]["kp"],
            current_kp, match_max_distance, draw=True)

        vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), current_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # determine median distance between all matched feature points
        median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
        print(median_dist)

        # insert a new key frame
        if median_dist >= 100.0:
            print("########## insert new KF ###########")

            # estimate KF pose
            R, t, mask = estimate_camera_pose(last_pts, current_pts, camera_matrix, min_inliers=20)

            R_last, t_last = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            R_current = np.matmul(R_last, R.T)
            t_current = t_last + -np.matmul(R.T, t.reshape(3,)).reshape(3,1)
            print("pose before scale correction: ", R_current, t_current)

            # filter inliers
            last_pts = last_pts[:, mask, :]
            current_pts = current_pts[:, mask, :]
            matches = list(np.array(matches)[mask])

            # triangulate map points
            R1, t1 = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            pts_3d = triangulate_map_points(last_pts, current_pts, R1, t1, R_current, t_current, camera_matrix)

            # rescale translation t by computing the distance ratio between triangulated world points and world points in previous keyframe
            pts_3d_prev, match_idxs = get_map_points_and_kps_for_matches(prev_node_id, matches)
            # find corresponding sub-array in current pts_3d
            pts_3d_current = pts_3d[match_idxs, :]

            # compute scale ratio for random pairs of points
            num_points = np.minimum(10000, pts_3d_prev.shape[0]**2)
            scale_ratios = []
            for _ in range(num_points):
                first_pt_idx, second_pt_idx = np.random.choice(range(pts_3d_prev.shape[0]), size=(2,), replace=False)
                # compute distance between the selected points
                dist_prev = np.linalg.norm(pts_3d_prev[first_pt_idx, :] - pts_3d_prev[second_pt_idx, :])
                dist_current = np.linalg.norm(pts_3d_current[first_pt_idx, :] - pts_3d_current[second_pt_idx, :])
                scale_ratio = dist_prev / dist_current
                scale_ratios.append(scale_ratio)

            #print("scale_ratios: ", scale_ratios, np.mean(scale_ratios), np.std(scale_ratios), np.median(scale_ratios))

            # rescale translation using the median scale ratio
            t *= np.median(scale_ratios)

            R_last, t_last = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            R_current = np.matmul(R_last, R.T)
            t_current = t_last + -np.matmul(R.T, t.reshape(3,)).reshape(3,1)
            print("pose after scale correction: ", R_current, t_current)

            # triangulate map points using the scale-corrected pose
            pts_3d = triangulate_map_points(last_pts, current_pts, R1, t1, R_current, t_current, camera_matrix)

            # insert new keyframe into pose graph
            pose_graph.add_node(prev_node_id+1,
                frame=current_frame,
                kp=current_kp,
                des=current_des,
                pose=to_twist(R_current, t_current))
            pose_graph.add_edge(prev_node_id, prev_node_id+1, num_matches=len(matches))

            # add new map points
            associated_kp_indices = [[m.queryIdx, m.trainIdx] for m in matches]
            map_points.insert(pts_3d, associated_kp_indices, observing_kfs=[prev_node_id, prev_node_id+1])
            print("pts_3d.mean: ", np.median(pts_3d, axis=0))


        cv2.imshow("current_frame", vis_current_frame)
        prev_node_id = sorted(pose_graph.nodes)[-1]
        cv2.imshow("last_keyframe", pose_graph.nodes[prev_node_id]["frame"])
        cv2.imshow("match_frame", match_frame)

        # handle key presses
        # 'q' - Quit the running program
        # 's' - enter stepwise mode
        # 'a' - exit stepwise mode
        key = cv2.waitKey(1)
        if not step_wise and key == ord('s'):
            step_wise = True
        if key == ord('q'):
            break
        if step_wise:
            while True:
                key = cv2.waitKey(1)
                if key == ord('s'):
                    break
                elif key == ord('a'):
                    step_wise = False
                    break

    # if any error occurs store result
    except:
        dump_result()
        print("Error occured. Dumped result.")
        raise

cv2.destroyAllWindows()

dump_result()
