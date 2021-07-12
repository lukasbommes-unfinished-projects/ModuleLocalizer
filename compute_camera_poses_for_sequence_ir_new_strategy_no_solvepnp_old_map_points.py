import os
import glob
import pickle
import numpy as np
import cv2
import networkx as nx
import g2o

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
    pose_graph.nodes[0]["kp_matched"] = last_pts.reshape(-1, 2)
    pose_graph.nodes[1]["kp_matched"] = current_pts.reshape(-1, 2)

    # compute relative camera pose for second frame
    essential_mat, _ = cv2.findEssentialMat(last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, method=cv2.LMEDS)  # RANSAC fails here
    num_inliers, R, t, mask = cv2.recoverPose(essential_mat, last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix)
    mask = mask.astype(np.bool).reshape(-1,)
    print(num_inliers)

    if num_inliers < 20:
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
    map_points = np.vstack((map_points, pts_3d))

    # Store indices of map points belonging to KF0 and KF1 in pose graph node
    pose_graph.nodes[0]["visible_map_points"] = [range(0, pts_3d.shape[0])]
    pose_graph.nodes[1]["visible_map_points"] = [range(0, pts_3d.shape[0])]

    print("Initialization successful. Chose frames 0 and {} as key frames".format(frame_idx_init))

    # TODO: perform full BA to optimize initial camera poses and map points [see. ORB_SLAM IV. 5)]

    return pose_graph, map_points, frame_idx_init


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

        # # retrieve map points visible by last key frame
        # prev_node_id = sorted(pose_graph.nodes)[-1]
        # visible_map_points = pose_graph.nodes[prev_node_id]["visible_map_points"]  # may contain multiple disconnected index ranges
        # try:
        #     pts_3d = np.vstack([map_points[vs, :] for vs in visible_map_points])
        # except ValueError:
        #     raise ValueError ("Last keyframe did not contain any visible map points.")

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

            ### Step 1: Estimate KF pose by decomposing essential matrix computed from 2D-2D point correspondences

            # Note: Tranlation t is only known up to scale
            essential_mat, mask = cv2.findEssentialMat(last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, method=cv2.LMEDS)
            num_inliers, R, t, mask = cv2.recoverPose(essential_mat, last_pts.reshape(1, -1, 2), current_pts.reshape(1, -1, 2), camera_matrix, mask=mask)
            mask = mask.astype(np.bool).reshape(-1,)
            print("recover pose num inliers: ", num_inliers)

            if num_inliers < 20:
                raise RuntimeError("Could not recover camera pose.")

            R_last_kf, t_last_kf = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            R_current = np.matmul(R_last_kf, R.T)
            t_current = t_last_kf + -np.matmul(R.T, t.reshape(3,)).reshape(3,1)
            current_pose = to_twist(R_current, t_current)
            print(R_current, t_current)

            # find matches of current frame with last key frame
            # triangulate new map points
            # insert map points into world map
            # insert key frame into pose graph

            R1, t1 = from_twist(pose_graph.nodes[prev_node_id]["pose"])
            R2, t2 = from_twist(current_pose)
            print("kf_candidate_pose", current_pose)

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

            # rescale translation t by computing the distance ratio between triangulated world points and world points in previous keyframe



            # insert new keyframe into pose graph
            pose_graph.add_node(prev_node_id+1,
                frame=current_frame,
                kp=current_kp,
                kp_matched=current_pts.reshape(-1, 2),
                des=current_des,
                pose=current_pose)
            pose_graph.add_edge(prev_node_id, prev_node_id+1, matches=matches)

            # add new map points to map point array and store index range of these new points in pose graph
            map_len = map_points.shape[0]
            map_points = np.vstack((map_points, pts_3d))
            pose_graph.nodes[prev_node_id]["visible_map_points"].append(range(map_len, map_len + pts_3d.shape[0]))  # triangulatde points are also visible from previous key frame
            pose_graph.nodes[prev_node_id+1]["visible_map_points"] = [range(map_len, map_len + pts_3d.shape[0])]
            print("pts_3d.mean: ", np.median(pts_3d, axis=0))

            ###################################################################
            #
            # local bundle adjustment
            #
            ###################################################################

            # ## setup optimizer and camera parameters
            # robust_kernel = True
            # optimizer = g2o.SparseOptimizer()
            # solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
            # solver = g2o.OptimizationAlgorithmLevenberg(solver)
            # optimizer.set_algorithm(solver)
            #
            # focal_length = (camera_matrix[0,0] + camera_matrix[1,1]) / 2
            # principal_point = (camera_matrix[0,2], camera_matrix[1,2])
            # print("focal_length: ", focal_length, "principal_point: ", principal_point)
            # cam = g2o.CameraParameters(focal_length, principal_point, 0)
            # cam.set_id(0)
            # optimizer.add_parameter(cam)
            #
            # # add current keyframe poses
            # poses = []
            # for i, node_id in enumerate(sorted(pose_graph.nodes)):
            #     R, t = from_twist(pose_graph.nodes[node_id]["pose"])
            #     pose = g2o.SE3Quat(R, np.squeeze(t))
            #     # problems: (fixed)
            #     # my twist coordinates are first rotation, then translation
            #     # Se3Quat minimal vector is first translation, then rotation
            #     # rotation seems to be encoded differently in g2o minimal vector
            #     poses.append(pose)
            #
            #     v_se3 = g2o.VertexSE3Expmap()
            #     print("pose i ", i)
            #     v_se3.set_id(i)
            #     v_se3.set_estimate(pose)
            #     #if i < 2:
            #     #    v_se3.set_fixed(True)
            #     optimizer.add_vertex(v_se3)
            #
            # # add map points
            # point_id = len(poses)
            # inliers = dict()
            # for i, point in enumerate(map_points):
            #     visible = []
            #     for j, pose in enumerate(poses):
            #         z = cam.cam_map(pose * point)
            #         if 0 <= z[0] < 640 and 0 <= z[1] < 512:
            #             visible.append((j, z))
            #     if len(visible) < 2:
            #         continue
            #
            #     vp = g2o.VertexSBAPointXYZ()
            #     vp.set_id(point_id)
            #     vp.set_marginalized(True)
            #     vp.set_estimate(point)
            #     optimizer.add_vertex(vp)
            #
            #     for j, z in visible:
            #         edge = g2o.EdgeProjectXYZ2UV()
            #         edge.set_vertex(0, vp)
            #         edge.set_vertex(1, optimizer.vertex(j))
            #         edge.set_measurement(z)
            #         edge.set_information(np.identity(2))
            #         if robust_kernel:
            #             edge.set_robust_kernel(g2o.RobustKernelHuber())
            #             #edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))  # 95% CI
            #
            #         edge.set_parameter_id(0, 0)
            #         optimizer.add_edge(edge)
            #
            #     inliers[point_id] = i
            #     point_id += 1
            #
            # print('num vertices:', len(optimizer.vertices()))
            # print('num edges:', len(optimizer.edges()))
            #
            # print('Performing full BA:')
            # optimizer.initialize_optimization()
            # optimizer.set_verbose(True)
            # optimizer.optimize(10)
            #
            # # # read out optimized poses
            # for i in range(len(poses)):
            #     vp = optimizer.vertex(i)
            #     se3quat = vp.estimate()
            #     R = np.copy(se3quat.to_homogeneous_matrix()[0:3, 0:3])
            #     t = np.copy(se3quat.to_homogeneous_matrix()[0:3, 3])
            #     pose_graph.nodes[i]["pose"] = to_twist(R, t)
            #
            # # read out optimized map points
            # for i, point_id in enumerate(inliers):
            #     vp = optimizer.vertex(point_id)
            #     map_points[i, :] = np.copy(vp.estimate())


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
