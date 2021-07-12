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


# camera parameters
# w = 1920
# h = 1080
# fx = 1184.51770
# fy = 1183.63810
# cx = 978.30778
# cy = 533.85598
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# dist_coeffs = np.array([-0.01581, 0.01052, -0.00075, 0.00245, 0.00000])

# read video
# video_file = "phantom3-village-original/flight_truncated.MOV"
# cap = cv2.VideoCapture(video_file)

# # precompute undistortion maps
# new_camera_matrix = camera_matrix
# #new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)
# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
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


# for testing remove the first part of the video where drone ascends
#[cap.read() for _ in range(1000)]

# TODO: It is a good idea to normalize the frame sbefore performing any operation
#  this helps to account for changes in lighting, exposure, etc.


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
    essential_mat, _ = cv2.findEssentialMat(last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, method=cv2.LMEDS)  # RANSAC fails here
    num_inliers, R, t, mask = cv2.recoverPose(essential_mat, last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix)
    mask = mask.astype(np.bool).reshape(-1,)
    print(num_inliers)

    if num_inliers < 50:
        raise RuntimeError("Could not recover intial camera pose based on selected keyframes. Insufficient parallax or number of feature points.")

    print("init R", R)
    print("init t", t)

    # relative camera pose
    R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float64)
    t1 = np.array([[0], [0], [0]]).astype(np.float64)
    R2 = R.T
    t2 = -np.matmul(R.T, t.reshape(3,)).reshape(3,1)

    # insert pose (in twist coordinates) of KF0 and KF1 into keyframes dict
    # poses are w.r.t. KF0 which is the base coordinate system of the entire map
    pose_graph.nodes[0]["pose"] = to_twist(R1, t1)
    pose_graph.nodes[1]["pose"] = to_twist(R2, t2)

    # create projection matrices needed for triangulation of initial 3D point cloud
    proj_matrix1 = np.hstack([R1.T, -R1.T.dot(t1)])
    proj_matrix2 = np.hstack([R2.T, -R2.T.dot(t2)])
    proj_matrix1 = camera_matrix.dot(proj_matrix1)
    proj_matrix2 = camera_matrix.dot(proj_matrix2)

    # triangulate initial 3D point cloud
    pts_3d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, last_pts.reshape(-1, 2).T, current_pts.reshape(-1, 2).T).T
    pts_3d = cv2.convertPointsFromHomogeneous(pts_3d).reshape(-1, 3)

    # filter outliers based on mask from recoverPose
    #pts_3d = pts_3d[valid_map_points_mask, :].reshape(-1, 3)

    # add triangulated points to map points
    associated_kp_indices = [[m.queryIdx, m.trainIdx] for m in matches]
    map_points.insert(pts_3d, associated_kp_indices, observing_kfs=[0, 1])  # triangulatedmap points between KF0 and KF1

    # Store indices of map points belonging to KF0 and KF1 in pose graph node
    #pose_graph.nodes[0]["visible_map_points"] = [range(0, pts_3d.shape[0])]
    #pose_graph.nodes[1]["visible_map_points"] = [range(0, pts_3d.shape[0])]

    print("Initialization successful. Chose frames 0 and {} as key frames".format(frame_idx_init))

    # TODO: perform full BA to optimize initial camera poses and map points [see. ORB_SLAM IV. 5)]

    return pose_graph, map_points, frame_idx_init


def estimate_camera_pose(img_points, pts_3d, camera_matrix):
    """Estimates the camera world pose of a frame based on 2D-3D
    corresponding points.

    Args:
        img_points (`numpy.ndarray`): A set of keypoints extracted from the
            current frame. Shape (-1, 1, 2). These keypoints can also be tracked
            from the previous frame.

        pts_3d (`numpy.ndarray`): Triangulated 3D points corresponding to
            the keypoints in img_points. Shape (-1, 1, 3). Note, that the order
            of the points in this array needs to be consistent with the order of
            keypoints in img_points.

        camera_matrix (`numpy.ndarray`): Camera matrix of the camera which
            was used to acquire frames.

    Returns:
        R (`numpy.ndarray`): Rotation matrix of the camera coordinate system
            w.r.t. world coordinate system. Shape (3, 3).
        t (`numpy.ndarray`): Translation (x, y, z) of the camera coordinate
            system w.r.t. world coordinate system. Shape (3,).

    Note:
        This function assumes keypoints to be extracted from an undistorted
        frame.
    """
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.reshape(-1, 1, 3),
        img_points.reshape(-1, 1, 2),
        camera_matrix,
        None,
        reprojectionError=1,
        iterationsCount=100,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("Could not compute the camera pose for the new frame with solvePnP.")
    print("solvePnP success", success)
    print("solvePnP inliers", inliers.shape)
    R = cv2.Rodrigues(rvec)[0].T
    t = -np.matmul(cv2.Rodrigues(rvec)[0].T, tvec)
    return R, t


def get_map_points_and_kps_for_matches(last_kf_index, matches, current_kp):
    """Returns map points and corresponding key points for current frame.

    Given matches between a current frame and the last keyframe the function
    finds which key point in the current frame correpsonds to which key point
    in the last key frame and returns the map points correpsodning to these
    key points. these can be used for solvePnP to get a first pose estimate of
    the current frame.
    """
    # get all map points observed in last KF
    _, pts_3d, associated_kp_indices = map_points.get_by_observation(last_kf_index)  # get all map points observed by last KF
    # get indices of map points which were found again in the current frame
    kp_idxs = []
    new_matches = []
    for m in matches:
        try:
            kp_idx = associated_kp_indices.index(m.queryIdx)
        except ValueError:
            pass
        else:
            kp_idxs.append(kp_idx)
            new_matches.append(m)
    print("{} of {} ({:3.3f} %) keypoints in last key frame have been found again in current frame".format(len(new_matches), len(matches), len(new_matches)/len(matches)))
    # get map points according to the indices
    pts_3d = pts_3d[np.array(kp_idxs), :]
    # get corresponding key points in the current frame
    img_points = np.array([current_kp[m.trainIdx].pt for m in new_matches]).reshape(-1, 1, 2)
    return pts_3d, img_points

################################################################################

pose_graph, map_points, frame_idx = initialize(fast, orb, camera_matrix)

# maintain a copy of the last frame's data so that we can use it to create a new
# keyframe once distance threshold is exceeded
kf_candidate_frame = None
kf_candidate_pose = None

last_frame_was_keyframe = False
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

        if not last_frame_was_keyframe:
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

        # get the map points and corresponding key points in last KF
        pts_3d, img_points = get_map_points_and_kps_for_matches(prev_node_id, matches, current_kp)

        # visualize which matches have been selected as image points
        w = current_frame.shape[1]
        for pt in img_points:
            match_frame = cv2.circle(match_frame, center=(int(w+pt[0, 0]), int(pt[0, 1])), radius=3, color=(0, 0, 0), thickness=-1)

        # recover initial camera pose of current frame by solving PnP
        print(img_points.shape)
        print(pts_3d)
        print(pts_3d.shape)
        print(pts_3d.dtype)
        #print("current_pts before PnP", img_points, "len", img_points.shape)
        R_current, t_current = estimate_camera_pose(img_points, pts_3d, camera_matrix)
        current_pose = to_twist(R_current, t_current)
        print(R_current, t_current)

        assert np.allclose(np.linalg.det(R_current), 1.0), "Determinant of rotation matrix in local tracking is not 1."

###

### local tracking (track local map and refine pose)


###

        # TODO: extend below: KF insertion decision based on travelled GPS distance and number of frames processed

        # decide when to insert a new keyframe based on a robust thresholding mechanism
        # the weights are chosen to make the difference more sensitive to changes in rotation and z-coordinate
        pose_distance_weights = np.array([[10, 0, 0, 0, 0, 0],  # r1
                                          [0, 10, 0, 0, 0, 0],  # r2
                                          [0, 0, 10, 0, 0, 0],  # r3
                                          [0, 0, 0, 1, 0, 0],   # t1 = x
                                          [0, 0, 0, 0, 1, 0],   # t2 = y
                                          [0, 0, 0, 0, 0, 10]]) # t3 = z
        pose_distance_threshold = 10
        # compute relative pose to pose of last keyframe
        prev_node_id = sorted(pose_graph.nodes)[-1]
        R_last_kf, t_last_kf = from_twist(pose_graph.nodes[prev_node_id]["pose"])
        R_delta = np.matmul(R_last_kf.T, R_current)
        t_delta = t_current - t_last_kf
        current_pose_delta = to_twist(R_delta, t_delta).reshape(6, 1)
        current_dist = np.matmul(np.matmul(current_pose_delta.T, pose_distance_weights), current_pose_delta)
        print(current_dist)

        if current_dist < pose_distance_threshold:
            last_frame_was_keyframe = False

            # maintain frame data in case the next frame exceeds the pose distance threshold
            kf_candidate_frame = current_frame
            kf_candidate_pose = current_pose

            # only store pose when it is accurately estimated
            Rs.append(R_current)
            ts.append(t_current)

        else:  # insert a new keyframe with data of previous frame, then track again
            last_frame_was_keyframe = True  # do not retrieve a new frame in the next iteration as we first need to process the already retrieved frame

            print("########## insert new KF ###########")

            # find matches of current frame with last key frame
            # triangulate new map points
            # insert map points into world map
            # insert key frame into pose graph

            # extract keypoints of new key frame and match with previous key frame
            kp_kf_match, des_kf_match = extract_kp_des(kf_candidate_frame, fast, orb)
            prev_node_id = sorted(pose_graph.nodes)[-1]
            matches, last_pts, current_pts, match_frame = match(bf,
                pose_graph.nodes[prev_node_id]["frame"],
                kf_candidate_frame,
                pose_graph.nodes[prev_node_id]["des"],
                des_kf_match, pose_graph.nodes[prev_node_id]["kp"],
                kp_kf_match, match_max_distance, draw=True)

            R1, t1 = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            R2, t2 = from_twist(kf_candidate_pose)
            print("kf_candidate_pose", kf_candidate_pose)

            print("R1, t1", (R1, t1))
            print("R2, t2", (R2, t2))

            # create projection matrices needed for triangulation of 3D points
            proj_matrix1 = np.hstack([R1.T, -R1.T.dot(t1)])
            proj_matrix2 = np.hstack([R2.T, -R2.T.dot(t2)])
            proj_matrix1 = camera_matrix.dot(proj_matrix1)
            proj_matrix2 = camera_matrix.dot(proj_matrix2)

            # triangulate new map points based on matches with previous key frame
            pts_3d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, last_pts.reshape(-1, 2).T, current_pts.reshape(-1, 2).T).T
            pts_3d = cv2.convertPointsFromHomogeneous(pts_3d).reshape(-1, 3)

            print("pts_3d", pts_3d)
            #map_points.append(pts_3d)

            # insert new keyframe into pose graph
            pose_graph.add_node(prev_node_id+1,
                frame=kf_candidate_frame,
                kp=kp_kf_match,
                des=des_kf_match,
                pose=kf_candidate_pose)
            pose_graph.add_edge(prev_node_id, prev_node_id+1, num_matches=len(matches))

            # add new map points
            associated_kp_indices = [[m.queryIdx, m.trainIdx] for m in matches]
            map_points.insert(pts_3d, associated_kp_indices, observing_kfs=[prev_node_id, prev_node_id+1])

            # remove points with negative z coordinate
            #pts_3d = pts_3d[np.where(pts_3d[:, 2] >= t1[2]), :].reshape(-1, 3)

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

cap.release()
cv2.destroyAllWindows()

dump_result()
