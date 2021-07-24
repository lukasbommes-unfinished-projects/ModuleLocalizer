import os
import glob
import json
import pickle
from collections import defaultdict, Counter
import numpy as np
import cv2
import networkx as nx
from scipy.spatial import KDTree

from mapper.map_points import MapPoints, get_representative_orb
from mapper.pose_graph import get_neighbors
from mapper.common import Capture, get_visible_points, \
    get_map_points_and_kps_for_matches, get_local_map
from mapper.keypoints import extract_keypoints, match
from mapper.geometry import from_twist, to_twist, estimate_camera_pose, \
    triangulate_map_points
from mapper.bundle_adjustment import bundle_adjust

# TODO:
# try out FLANN matcher with epipolar contraint to speed up ORB matching and reduce outliers
# see: https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html

# TODO: if system fails due to low number of matches, restart and use last known
#       pose as initial pose


def get_frame(cap):
    """Reads and undistorts next frame from stream."""
    frame, _, frame_name, _ = cap.get_next_frame(
        preprocess=True, undistort=True, equalize_hist=True)
    return frame, frame_name


def dump_result(path="."):
    pickle.dump(map_points, open(os.path.join(path, "map_points.pkl"), "wb"))
    for node in pose_graph.nodes:
        pose_graph.nodes[node]["kp"] = cv2.KeyPoint_convert(pose_graph.nodes[node]["kp"])
    pickle.dump(pose_graph, open(os.path.join(path, "pose_graph.pkl"), "wb"))


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
                             # edges stores neighbor relationships between keyframes
    map_points = MapPoints()  # stores 3D world points

    # get first key frame
    frame, frame_name = get_frame(cap)
    #kp, des = extract_keypoints(frame, fast, orb)
    kp, des = extract_keypoints(frame, orb)
    pose_graph.add_node(0, frame=frame, frame_name=frame_name, kp=kp, des=des)

    frame_idx_init = 0

    while True:
        frame, frame_name = get_frame(cap)
        if frame is None:
            break
        frame_idx_init += 1

        # extract keypoints and match with first key frame
        #kp, des = extract_keypoints(frame, fast, orb)
        kp, des = extract_keypoints(frame, orb)
        matches, last_pts, current_pts, match_frame = match(bf,
            pose_graph.nodes[0]["frame"], frame, pose_graph.nodes[0]["des"],
            des, pose_graph.nodes[0]["kp"], kp, match_max_distance, draw=False)

        # determine median distance between all matched feature points
        median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
        print(median_dist)

        # if distance exceeds threshold choose frame as second keyframe
        if median_dist >= min_parallax or len(matches) < 200:
            pose_graph.add_node(1, frame=frame, frame_name=frame_name, kp=kp, des=des)
            break

    # compute relative camera pose for second frame
    R, t, mask = estimate_camera_pose(last_pts, current_pts, camera_matrix, min_inliers=20)

    # filter inliers
    last_pts = last_pts[:, mask, :]
    current_pts = current_pts[:, mask, :]
    matches = list(np.array(matches)[mask])

    pose_graph.add_edge(0, 1, num_matches=len(matches))

    # relative camera pose
    R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float64)
    t1 = np.array([[0], [0], [0]]).astype(np.float64)
    R2 = R.T
    t2 = -np.matmul(R.T, t.reshape(3,)).reshape(3,1)

    # insert pose (in twist coordinates) of KF0 and KF1 into keyframes dict
    # poses are w.r.t. KF0 which is the base coordinate system of the entire map
    pose_graph.nodes[0]["pose"] = to_twist(R1, t1)
    pose_graph.nodes[1]["pose"] = to_twist(R2, t2)

    # update boolean flags indicating which of the keypoints where matched
    # NOT NEEDED
    #update_matched_flag(pose_graph, matches)

    # triangulate initial 3D point cloud
    pts_3d = triangulate_map_points(last_pts, current_pts, R1, t1, R2, t2, camera_matrix)

    # filter outliers
    #pts_3d = remove_outliers(pts_3d)

    # add triangulated points to map points
    map_points.insert(
        pts_3d,
        associated_kp_indices=[[m.queryIdx, m.trainIdx] for m in matches],
        representative_orb=[des[m.trainIdx, :] for m in matches],
        observing_kfs=[[0, 1] for _ in matches]
    )

    print("Initialization successful. Chose frames 0 and {} as key frames".format(frame_idx_init))

    print("Num map points after init: {}".format(map_points.pts_3d.shape))

    # TODO: perform full BA to optimize initial camera poses and map points [see. ORB_SLAM IV. 5)]

    return pose_graph, map_points, frame_idx_init


def insert_keyframe(pose_graph, map_points, last_pts, current_pts, matches,
    current_kp, current_des, frame, frame_name, camera_matrix):
    """Insert a new keyframe.

    Once the median distance of matched keypoints between the
    current frame and last key frame exceeds a threshold, the
    current frame is inserted as a new key frame. Its pose relative
    to the previous frame is computed by decomposing the essential
    matrix computed from the matched points. This is a pose
    estimation from 2D-2D point correspondences and as such does
    not suffer from inaccuricies in the triangulated map points.
    Map points are triangulated between the new key frame and the
    last key frame.
    As the scale of the translation is only known up to a constant
    factor, we compute the scale ratio from as quotient of Euclidean
    distance between corresponding pairs of map points in the last
    and current key frame. For robustness, we sample many pairs and
    use the median of estimated scale ratios.
    Once the actual scale is known, the camera pose is updated and
    map points are triangulated again. Camera pose and map points
    are then inserted in the pose graph and map points object.
    """
    # estimate KF pose
    R, t, mask = estimate_camera_pose(
        last_pts, current_pts, camera_matrix, min_inliers=20)

    prev_node_id = sorted(pose_graph.nodes)[-1]
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
    pts_3d = triangulate_map_points(
        last_pts, current_pts, R1, t1, R_current, t_current, camera_matrix)

    # rescale translation t by computing the distance ratio between
    # triangulated world points and world points in previous keyframe
    pts_3d_prev, match_idxs = get_map_points_and_kps_for_matches(
        map_points, prev_node_id, matches)
    # find corresponding sub-array in current pts_3d
    pts_3d_current = pts_3d[match_idxs, :]

    # compute scale ratio for random pairs of points
    num_points = np.minimum(10000, pts_3d_prev.shape[0]**2)
    scale_ratios = []
    for _ in range(num_points):
        first_pt_idx, second_pt_idx = np.random.choice(
            range(pts_3d_prev.shape[0]), size=(2,), replace=False)
        # compute distance between the selected points
        dist_prev = np.linalg.norm(
            pts_3d_prev[first_pt_idx, :] - pts_3d_prev[second_pt_idx, :])
        dist_current = np.linalg.norm(
            pts_3d_current[first_pt_idx, :] - pts_3d_current[second_pt_idx, :])
        scale_ratio = dist_prev / dist_current
        scale_ratios.append(scale_ratio)

    # rescale translation using the median scale ratio
    t *= np.median(scale_ratios)

    R_last, t_last = from_twist(pose_graph.nodes[prev_node_id]["pose"])
    R_current = np.matmul(R_last, R.T)
    t_current = t_last + -np.matmul(R.T, t.reshape(3,)).reshape(3,1)
    print("pose after scale correction: ", R_current, t_current)

    # triangulate map points using the scale-corrected pose
    pts_3d = triangulate_map_points(
        last_pts, current_pts, R1, t1, R_current, t_current, camera_matrix)

    # insert new keyframe into pose graph
    pose_graph.add_node(prev_node_id+1,
        frame=frame,
        frame_name=frame_name,
        kp=current_kp,
        des=current_des,
        pose=to_twist(R_current, t_current))
    pose_graph.add_edge(prev_node_id, prev_node_id+1, num_matches=len(matches))

    # update boolean flags indicating which of the keypoints where matched
    # NOT NEEDED
    #update_matched_flag(pose_graph, matches)

    # add new map points
    map_points.insert(
        pts_3d,
        associated_kp_indices=[[m.queryIdx, m.trainIdx] for m in matches],
        representative_orb=[current_des[m.trainIdx, :] for m in matches],
        observing_kfs=[[prev_node_id, prev_node_id+1] for _ in matches]
    )
    print("pts_3d.mean: ", np.median(pts_3d, axis=0))

    print("Num map points after inserting KF: {}".format(map_points.pts_3d.shape))


def find_neighbor_keyframes(pose_graph, map_points, frame, camera_matrix,
    min_shared_points=40):
    """Find neighboring keyframes.

    Adds edges between the newest keyframe and other keyframes
    sharing enough map points.

    Project the most recently added map points into all key
    frames in the pose graph. For each map point determine in
    which key frame(s) it is visible and add an entry into the
    observation dict of the map point (setting its key point
    index to None as no keypoint is matched yet). If two key
    frames share more than `min_shared_points` add an edge
    between them in the pose graph.
    """

    newest_node_id = sorted(pose_graph.nodes)[-1]
    newest_map_points_idxs, newest_map_points, _, _ = \
        map_points.get_by_observation(newest_node_id)

    for node_id in sorted(pose_graph.nodes):
        if (node_id >= newest_node_id-1):  # skip this and previous keyframe
            continue

        # project newest map points into each key frame
        R, t = from_twist(pose_graph.nodes[node_id]["pose"])
        projected_pts, _ = cv2.projectPoints(
            newest_map_points, R.T, -R.T.dot(t), camera_matrix, None)

        # filter out those which do not lie within the frame bounds
        projected_pts, mask = get_visible_points(projected_pts,
            frame_width=frame.shape[1], frame_height=frame.shape[0])

        # insert this key frame (node_id) into map point
        # observations of all visible map points
        for idx, m in zip(newest_map_points_idxs, mask):
            if m:
                map_points.observations[idx][node_id] = None

        # if enough map points are visible insert edge in pose graph
        if len(projected_pts) > min_shared_points:
            print("Key frame {} shares {} map points with key frame {}".format(
                newest_node_id, len(projected_pts), node_id))
            pose_graph.add_edge(
                newest_node_id, node_id, num_matches=len(projected_pts))


def update_map_oberservations(map_points, pose_graph, camera_matrix,
    max_search_radius=16.0, match_max_distance=20.0):
    """Finds key points corresponding to current map points in other key frames.

    Args:
        max_search_radius (`float`): Associate map points with a keypoint only
            if the keypoint is at most this distance ayway form the projected
            point.

        match_max_distance (`float`): Maximum distance (not spatial distance,
            but match distance) for a match to be considered valid.

    For last keyframe:
        1) Obtain a local map, i.e. all map points visible in the
           last keyframe and its neighbouring keyframes
        2) Project map point into each keyframe using the current
           keyframe pose
        3) Search in a local region around the projected point for
           ORB descriptors (using the representative ORB descriptor
           of the map point)
        4) If a match is found, add the corresponding keyframe to
           the observation dict of that map point
        5) Update the representative ORB descriptor of that map point
    """

    newest_node_id = list(sorted(pose_graph.nodes))[-1]
    print("Data association for keyframe {}".format(newest_node_id))

    # get node_ids of neighboring keyframes
    neighbors_keyframes = get_neighbors(pose_graph, newest_node_id)
    print("Neighboring keyframes: {}".format(neighbors_keyframes))

    # obtain map points visible in the neighbouring key frames
    map_points_local = get_local_map(map_points, neighbors_keyframes)

    descriptors = defaultdict(list)
    for node_id in neighbors_keyframes:
        # project map points into each key frame
        R, t = from_twist(pose_graph.nodes[node_id]["pose"])
        projected_pts, _ = cv2.projectPoints(map_points_local.pts_3d,
            R.T, -R.T.dot(t), camera_matrix, None)

        # for each projected point visible in KF[node_id]
        # search for matches with keypoints in local neighborhod of projected point
        # if match could be found update the visible KF and associated kp indices of the corresponding map point
        kp = pose_graph.nodes[node_id]["kp"]
        des = pose_graph.nodes[node_id]["des"]

        # build a mask for ORB descriptor matching which permits only matches of nearby points
        mask = np.zeros((len(map_points_local.representative_orb), len(des)), dtype=np.uint8)
        kp_ = cv2.KeyPoint_convert(kp).astype(np.uint16)
        pts = projected_pts.reshape(-1, 2)
        for i in range(len(pts)):
            d = (kp_[:, 0] - pts[i, 0])**2 + (kp_[:, 1] - pts[i, 1])**2
            mask[i, d < max_search_radius**2] = 1

        # find matches between projected map points and descriptors
        bf_local = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # set to False
        matches = bf_local.match(map_points_local.representative_orb, des, mask)  # TODO: select only those map points of the local group
        # filter out matches with distance (descriptor appearance) greater than threshold
        matches = [m for m in matches if m.distance < match_max_distance]
        print("Found {} new matches".format(len(matches)))

        for m in matches:
            # m.queryIdx: index of projected map point
            # m.trainIdx: index of keypoint in current KF
            map_point_idx_local = m.queryIdx
            map_point_idx = map_points_local.idx[map_point_idx_local]  # transform idx from local to global map
            kp_idx = m.trainIdx

            # store descriptor for later merging
            descriptors[map_point_idx].append(des[kp_idx])

            # update observing keyframes and associated keypoint indices
            map_points.observations[map_point_idx][node_id] = kp_idx

    # update representative ORB descriptor of each map points
    for map_point_idx, des in descriptors.items():
        if len(des) < 2:
            continue
        map_points.representative_orb[map_point_idx, :] = get_representative_orb(des)

    # print stats
    print(Counter(sorted([len([v for v in ob.values() if v is not None]) for ob in map_points.observations])))
    print(Counter(sorted([len(ob) for ob in map_points.observations])))


def local_bundle_adjustment(map_points, pose_graph, camera_matrix, keypoint_scale_levels):
    """Perform local bundle adjustment with local map and local keyframes."""
    newest_node_id = list(sorted(pose_graph.nodes))[-1]
    neighbors_keyframes = get_neighbors(pose_graph, newest_node_id)
    nodes = [*neighbors_keyframes, newest_node_id]
    print("Bundle adjustment for keyframe {}".format(newest_node_id))
    print("Neighboring keyframes: {}".format(neighbors_keyframes))
    # set robust kernel to 95 % confidence interval of local map
    map_points_local = get_local_map(map_points, neighbors_keyframes)
    robust_kernel_value = np.sqrt(5.99) #1.96*np.std(map_points_local.pts_3d)
    # perform local bundle adjustment
    bundle_adjust(pose_graph, map_points, nodes, camera_matrix,
        keypoint_scale_levels, robust_kernel_value)



if __name__ == "__main__":
    camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
    dist_coeffs = pickle.load(open("camera_calibration/parameters/ir/dist_coeffs.pkl", "rb"))

    frames_root = "data_processing/splitted"
    frame_files = sorted(glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
    #frame_files = frame_files[1410:] #[10094:] #[18142:] #[11138:]
    cap = Capture(frame_files, None, camera_matrix, dist_coeffs)

    gps_file = "data_processing/splitted/gps/gps.json"
    gps = json.load(open(gps_file, "r"))

    #orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
    orb_scale_factor = 1.2
    orb_nlevels = 8
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=12, scaleFactor=orb_scale_factor, nlevels=orb_nlevels)
    keypoint_scale_levels = np.array([orb_scale_factor**i for i in range(orb_nlevels)])
    match_max_distance = 20.0

    cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("match_frame", 1600, 900)

    step_wise = True
    match_frame = None

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

            frame, frame_name = get_frame(cap)
            if frame is None:
                break
            frame_idx += 1
            print("frame", frame_idx)

            # get initial pose estimate by matching keypoints with previous KF
            #current_kp, current_des = extract_keypoints(frame, fast, orb)
            current_kp, current_des = extract_keypoints(frame, orb)
            prev_node_id = sorted(pose_graph.nodes)[-1]
            matches, last_pts, current_pts, match_frame = match(bf,
                pose_graph.nodes[prev_node_id]["frame"],
                frame,
                pose_graph.nodes[prev_node_id]["des"],
                current_des, pose_graph.nodes[prev_node_id]["kp"],
                current_kp, match_max_distance, draw=True)

            # determine median distance between all matched feature points
            median_dist = np.median(
                np.linalg.norm(last_pts.reshape(-1, 2) -
                current_pts.reshape(-1, 2), axis=1))
            print("Median spatial distance of matches: {}".format(median_dist))

            if median_dist >= 100.0 or len(matches) < 200:

                print("########## insert new KF ###########")
                insert_keyframe(pose_graph, map_points, last_pts,
                    current_pts, matches, current_kp, current_des, frame,
                    frame_name, camera_matrix)

                find_neighbor_keyframes(pose_graph, map_points, frame,
                   camera_matrix)

                print("########## performing data association ###########")
                update_map_oberservations(map_points, pose_graph, camera_matrix)

                print("########## performing local bundle adjustment ###########")
                local_bundle_adjustment(map_points, pose_graph, camera_matrix, keypoint_scale_levels)


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
