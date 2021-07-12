import os
import pickle
import numpy as np
import cv2
import networkx as nx

from ssc import ssc

# camera parameters
w = 4000
h = 3000
camera_matrix = pickle.load(open("camera_calibration/parameters/rgb/camera_matrix.pkl", "rb"))
dist_coeffs = pickle.load(open("camera_calibration/parameters/rgb/dist_coeffs.pkl", "rb"))

# read video
video_file = "/storage/data/raw/20210510_Schmalenbach/02_north/VIS/DJI_0002.mov"
cap = cv2.VideoCapture(video_file)

# precompute undistortion maps
new_camera_matrix = camera_matrix
#new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
fast = cv2.FastFeatureDetector_create(threshold=12)
num_ret_points = 3000
tolerance = 0.1

lk_params = dict( winSize  = (21, 21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_kp_des(frame, fast, orb):
    kp = fast.detect(frame, None)
    kp = sorted(kp, key = lambda x:x.response, reverse=True)
    kp = ssc(kp, num_ret_points, tolerance, frame.shape[1], frame.shape[0])
    kp, des = orb.compute(frame, kp)
    # good features to track lead to cleaner tracks, but much more noisy pose estimates
    #kp = cv2.goodFeaturesToTrack(frame, **feature_params)
    #kp = cv2.KeyPoint_convert(kp)
    #kp, des = orb.compute(frame, kp)
    return kp, des


def match(bf, last_keyframe, current_frame, last_des, des, last_kp, kp, distance_threshold=25.0, draw=True):
    matches = bf.match(last_des, des)
    matches = sorted(matches, key = lambda x:x.distance)
    # filter out matches with distance (descriptor appearance) greater than threshold
    matches = [m for m in matches if m.distance < distance_threshold]
    print("Found {} matches of current frame with last frame".format(len(matches)))
    last_pts = np.array([last_kp[m.queryIdx].pt for m in matches]).reshape(1, -1, 2)
    current_pts = np.array([kp[m.trainIdx].pt for m in matches]).reshape(1, -1, 2)
    match_frame = np.zeros_like(current_frame)
    if draw:
        match_frame = cv2.drawMatches(last_keyframe, last_kp, current_frame, kp, matches, None, matchColor=(0, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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


def get_frame(cap, mapx, mapy):
    """Reads and undistorts next frame from stream."""
    retc, frame = cap.read()
    if not retc:
        raise RuntimeError("Could not read the first camera frame.")
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)  # undistort frame
    return frame


def gray(frame):
    """Convert BGR frame to gray frame."""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame_gray


def dump_result(path="."):
    pickle.dump(Rs, open(os.path.join(path, "Rs.pkl"), "wb"))
    pickle.dump(ts, open(os.path.join(path, "ts.pkl"), "wb"))
    pickle.dump(map_points, open(os.path.join(path, "map_points.pkl"), "wb"))

    # extract keyframe poses and visible map points from pose graph for plotting
    kf_poses = [data["pose"] for _, data in pose_graph.nodes.data()]
    pickle.dump(kf_poses, open(os.path.join(path, "kf_poses.pkl"), "wb"))
    kf_visible_map_points = [data["visible_map_points"] for _, data in pose_graph.nodes.data()]
    pickle.dump(kf_visible_map_points, open(os.path.join(path, "kf_visible_map_points.pkl"), "wb"))
    kf_frames = [data["frame"] for _, data in pose_graph.nodes.data()]
    pickle.dump(kf_frames, open(os.path.join(path, "kf_frames.pkl"), "wb"))
    kf_kp_matched = [data["kp_matched"] for _, data in pose_graph.nodes.data()]
    pickle.dump(kf_kp_matched, open(os.path.join(path, "kf_kp_matched.pkl"), "wb"))



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

frame_idx = 0


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
    map_points = np.empty(shape=(0, 3), dtype=np.float64)  # stores 3D world points
    #map_points = []

    # get first key frame
    frame = get_frame(cap, mapx, mapy)
    kp, des = extract_kp_des(gray(frame), fast, orb)
    pose_graph.add_node(0, frame=frame, kp=kp, des=des)

    frame_idx_init = 0

    while True:
        frame = get_frame(cap, mapx, mapy)

        # extract keypoints and match with first key frame
        kp, des = extract_kp_des(gray(frame), fast, orb)
        matches, last_pts, current_pts, match_frame = match(bf,
            gray(pose_graph.nodes[0]["frame"]), gray(frame), pose_graph.nodes[0]["des"],
            des, pose_graph.nodes[0]["kp"], kp, distance_threshold=25.0, draw=False)

        # determine median distance between all matched feature points
        median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
        print(median_dist)

        # if distance exceeds threshold choose frame as second keyframe
        if median_dist >= min_parallax:
            pose_graph.add_node(1, frame=frame, kp=kp, des=des)
            break

        frame_idx_init += 1

    pose_graph.add_edge(0, 1, matches=matches)

    # separately store the keypoints in matched order for tracking later
    pose_graph.nodes[0]["kp_matched"] = last_pts.reshape(-1, 2)
    pose_graph.nodes[1]["kp_matched"] = current_pts.reshape(-1, 2)

    # compute relative camera pose for second frame
    essential_mat, _ = cv2.findEssentialMat(last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, method=cv2.LMEDS)  # RANSAC fails here
    num_inliers, R, t, mask = cv2.recoverPose(essential_mat, last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix)
    mask = mask.astype(np.bool).reshape(-1,)
    print(num_inliers)

    if num_inliers >= 0.25*current_pts.reshape(1, -1, 2).shape[1]:
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
        map_points = np.vstack((map_points, pts_3d))  # map_points stores 3D points w.r.t. KF0
        #map_points.append(pts_3d)  # TODO: merge point clouds into single array, store in each kf in the pose graph which points are observable from this KF

        # Store indices of map points belonging to KF0 and KF1 in pose graph node
        pose_graph.nodes[0]["visible_map_points"] = [range(0, pts_3d.shape[0])]
        pose_graph.nodes[1]["visible_map_points"] = [range(0, pts_3d.shape[0])]

        print("Initialization successful. Chose frames 0 and {} as key frames".format(frame_idx_init))

    else:
        raise RuntimeError("Could not recover intial camera pose based on selected keyframes. Insufficient parallax or number of feature points.")

    # TODO: perform full BA to optimize initial camera poses and map points [see. ORB_SLAM IV. 5)]

    return pose_graph, map_points


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
    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d.reshape(-1, 1, 3), img_points.reshape(-1, 1, 2), camera_matrix, None, reprojectionError=8, iterationsCount=100)
    if not success:
        raise RuntimeError("Could not compute the camera pose for the new frame with solvePnP.")
    print("solvePnP success", success)
    print("solvePnP inliers", inliers.shape)
    R = cv2.Rodrigues(rvec)[0].T
    t = -np.matmul(cv2.Rodrigues(rvec)[0].T, tvec)
    return R, t



pose_graph, map_points = initialize(fast, orb, camera_matrix)

# needed for keypoint tracking
previous_frame = None
previous_kp = None

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

        print("frame", frame_idx)

        if frame_idx == 0:
            current_frame = get_frame(cap, mapx, mapy)
            # store frame data for next iteration
            previous_frame = current_frame
            prev_node_id = sorted(pose_graph.nodes)[-1]
            previous_kp = pose_graph.nodes[prev_node_id]["kp_matched"].reshape(1, -1, 2)
            vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), cv2.KeyPoint_convert(previous_kp), None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            frame_idx += 1
            continue

        if not last_frame_was_keyframe:
            current_frame = get_frame(cap, mapx, mapy)
            frame_idx += 1

        # track matched kps of last key frame
        print("performing tracking")
        p0 = np.float32(previous_kp).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(gray(previous_frame), gray(current_frame), p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(gray(current_frame), gray(previous_frame), p1, None, **lk_params)  # back-tracking
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        current_kp = p1
        vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), cv2.KeyPoint_convert(current_kp), None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # recover camera pose of current frame by solving PnP
        img_points = current_kp#[mask&good, :]  # 2D points in current frame
        #img_points = current_kp[valid_map_points_mask, :]#[mask&good, :]  # 2D points in current frame
        #pts_3d = map_points[-1]["pts_3d"]#[mask&good, :]  # corresponding 3D points in previous key frame
        #pts_3d = map_points[-1] # TODO: retrieve only those map point svisible by the key frames

        # retrieve map points visible by last key frame
        prev_node_id = sorted(pose_graph.nodes)[-1]
        visible_map_points = pose_graph.nodes[prev_node_id]["visible_map_points"]  # may contain multiple disconnected index ranges
        try:
            pts_3d = np.vstack([map_points[vs, :] for vs in visible_map_points])
        except ValueError:
            raise ValueError ("Last keyframe did not contain any visible map points.")

        print(pts_3d)
        print(pts_3d.shape)
        print(pts_3d.dtype)
        #print("current_pts before PnP", img_points, "len", img_points.shape)
        R_current, t_current = estimate_camera_pose(img_points, pts_3d, camera_matrix)
        current_pose = to_twist(R_current, t_current)
        print(R_current, t_current)

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

            # update for tracking of keypoints from previous to next frame
            previous_frame = current_frame
            previous_kp = current_kp

        else:  # insert a new keyframe with data of previous frame, then track again
            last_frame_was_keyframe = True  # do not retrieve a new frame in the next iteration as we first need to process the already retrieved frame

            print("########## insert new KF ###########")

            # find matches of current frame with last key frame
            # triangulate new map points
            # insert map points into world map
            # insert key frame into pose graph

            # extract keypoints of new key frame and match with previous key frame
            kp_kf_match, des_kf_match = extract_kp_des(gray(kf_candidate_frame), fast, orb)
            prev_node_id = sorted(pose_graph.nodes)[-1]
            matches, last_pts, current_pts, match_frame = match(bf,
                gray(pose_graph.nodes[prev_node_id]["frame"]),
                gray(kf_candidate_frame),
                pose_graph.nodes[prev_node_id]["des"],
                des_kf_match, pose_graph.nodes[prev_node_id]["kp"],
                kp_kf_match, distance_threshold=25.0, draw=False)

            R1, t1 = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            R2, t2 = from_twist(kf_candidate_pose)

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
                kp_matched=current_pts.reshape(-1, 2),
                des=des_kf_match,
                pose=kf_candidate_pose)
            pose_graph.add_edge(prev_node_id, prev_node_id+1, matches=matches)

            # add new map points to map point array and store index range of these new points in pose graph
            map_len = map_points.shape[0]
            map_points = np.vstack((map_points, pts_3d))
            pose_graph.nodes[prev_node_id]["visible_map_points"].append(range(map_len, map_len + pts_3d.shape[0]))  # triangulatde points are also visible from previous key frame
            pose_graph.nodes[prev_node_id+1]["visible_map_points"] = [range(map_len, map_len + pts_3d.shape[0])]

            # remove points with negative z coordinate
            #pts_3d = pts_3d[np.where(pts_3d[:, 2] >= t1[2]), :].reshape(-1, 3)

            # update for tracking of keypoints from previous to next frame
            previous_frame = kf_candidate_frame
            previous_kp = current_pts

        cv2.imshow("current_frame", vis_current_frame)
        prev_node_id = sorted(pose_graph.nodes)[-1]
        cv2.imshow("last_keyframe", pose_graph.nodes[prev_node_id]["frame"])

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
