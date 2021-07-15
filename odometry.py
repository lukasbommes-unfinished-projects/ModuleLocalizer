import os
import glob
import json
import pickle
import copy
from collections import defaultdict, Counter
import numpy as np
import cv2
import networkx as nx
import g2o
from scipy.spatial import KDTree

from mapper.map_points import MapPoints, get_representative_orb
from mapper.common import Capture, get_visible_points, update_matched_flag
from mapper.keypoints import extract_keypoints, match
from mapper.geometry import from_twist, to_twist, estimate_camera_pose, \
    triangulate_map_points

# TODO:
# try out FLANN matcher with epipolar contraint to speed up ORB matching and reduce outliers
# see: https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html


if __name__ == "__main__":
    camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
    dist_coeffs = pickle.load(open("camera_calibration/parameters/ir/dist_coeffs.pkl", "rb"))

    frames_root = "data_processing/splitted"
    frame_files = sorted(glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
    cap = Capture(frame_files, None, camera_matrix, dist_coeffs)

    gps_file = "data_processing/splitted/gps/gps.json"
    gps = json.load(open(gps_file, "r"))

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
    match_max_distance = 20.0


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
        #kf_poses = [data["pose"] for _, data in pose_graph.nodes.data()]
        # pickle.dump(kf_poses, open(os.path.join(path, "kf_poses.pkl"), "wb"))
        # kf_frames = [data["frame"] for _, data in pose_graph.nodes.data()]
        # pickle.dump(kf_frames, open(os.path.join(path, "kf_frames.pkl"), "wb"))
        # kf_frame_names = [data["frame_name"] for _, data in pose_graph.nodes.data()]
        # pickle.dump(kf_frame_names, open(os.path.join(path, "kf_frame_names.pkl"), "wb"))


    cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("match_frame", 1600, 900)

    step_wise = True
    match_frame = None


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
        kp, des = extract_keypoints(frame, fast, orb)
        pose_graph.add_node(0, frame=frame, frame_name=frame_name, kp=kp, des=des)

        frame_idx_init = 0

        while True:
            frame, frame_name = get_frame(cap)
            if frame is None:
                break
            frame_idx_init += 1

            # extract keypoints and match with first key frame
            kp, des = extract_keypoints(frame, fast, orb)
            matches, last_pts, current_pts, match_frame = match(bf,
                pose_graph.nodes[0]["frame"], frame, pose_graph.nodes[0]["des"],
                des, pose_graph.nodes[0]["kp"], kp, match_max_distance, draw=False)

            # determine median distance between all matched feature points
            median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
            print(median_dist)

            # if distance exceeds threshold choose frame as second keyframe
            if median_dist >= min_parallax:
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
        update_matched_flag(pose_graph, matches)

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


    def get_map_points_and_kps_for_matches(last_kf_index, matches):
        """Returns map points and corresponding key points for current frame.

        Given matches between a current frame and the last keyframe the function
        finds which key point in the current frame correpsonds to which key point
        in the last key frame and returns the map points correpsodning to these
        key points. It also returns the indices in the `matches` array corresponding
        to the returned 3D points.
        """
        # get all map points observed in last KF
        _, pts_3d, associated_kp_indices, _ = map_points.get_by_observation(last_kf_index)  # get all map points observed by last KF
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

            frame, frame_name = get_frame(cap)
            if frame is None:
                break

            frame_idx += 1

            print("frame", frame_idx)

            # get initial pose estimate by matching keypoints with previous KF
            current_kp, current_des = extract_keypoints(frame, fast, orb)
            prev_node_id = sorted(pose_graph.nodes)[-1]
            matches, last_pts, current_pts, match_frame = match(bf,
                pose_graph.nodes[prev_node_id]["frame"],
                frame,
                pose_graph.nodes[prev_node_id]["des"],
                current_des, pose_graph.nodes[prev_node_id]["kp"],
                current_kp, match_max_distance, draw=True)

            #vis_frame = cv2.drawKeypoints(np.copy(frame), current_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # determine median distance between all matched feature points
            median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
            print(median_dist)

            ###################################################################
            #
            # Insert new keyframe
            #
            #   Once the median distance of matched keypoints between the
            #   current frame and last key frame exceeds a threshold, the
            #   current frame is inserted as a new key frame. Its pose relative
            #   to the previous frame is computed by decomposing the essential
            #   matrix computed from the matched points. This is a pose
            #   estimation from 2D-2D point correspondences and as such does
            #   not suffer from inaccuricies in the triangulated map points.
            #   Map points are triangulated between the new key frame and the
            #   last key frame.
            #   As the scale of the translation is only known up to a constant
            #   factor, we compute the scale ratio from as quotient of Euclidean
            #   distance between corresponding pairs of map points in the last
            #   and current key frame. For robustness, we sample many pairs and
            #   use the median of estimated scale ratios.
            #   Once the actual scale is known, the camera pose is updated and
            #   map points are triangulated again. Camera pose and map points
            #   are then inserted in the pose graph and map points object.
            #
            ###################################################################

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
                    frame=frame,
                    frame_name=frame_name,
                    kp=current_kp,
                    des=current_des,
                    pose=to_twist(R_current, t_current))
                pose_graph.add_edge(prev_node_id, prev_node_id+1, num_matches=len(matches))

                # update boolean flags indicating which of the keypoints where matched
                # NOT NEEDED
                update_matched_flag(pose_graph, matches)

                # add new map points
                map_points.insert(
                    pts_3d,
                    associated_kp_indices=[[m.queryIdx, m.trainIdx] for m in matches],
                    representative_orb=[current_des[m.trainIdx, :] for m in matches],
                    observing_kfs=[[prev_node_id, prev_node_id+1] for _ in matches]
                )
                print("pts_3d.mean: ", np.median(pts_3d, axis=0))

                print("Num map points after inserting KF: {}".format(map_points.pts_3d.shape))

                ###################################################################
                #
                # Find neighboring keyframes
                #
                ###################################################################

                # add edges between the newest keyframe and other keyframes
                # sharing enough map points
                min_shared_points = 40  # if too low, then outliers will be projected also

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
                        print("Key frame {} shares {} map points with key frame {}".format(newest_node_id, len(projected_pts), node_id))
                        pose_graph.add_edge(
                            newest_node_id, node_id, num_matches=len(projected_pts))

                ###################################################################
                #
                # Data association
                #
                ###################################################################

                print("########## performing data association ###########")

                newest_node_id = list(sorted(pose_graph.nodes))[-1]
                print("Data association for keyframe {}".format(newest_node_id))

                # get node_ids of neighboring keyframes
                neighbors_keyframes = [node_id for _, node_id in sorted(
                    pose_graph.edges(newest_node_id))]
                print("Neighboring keyframes: {}".format(neighbors_keyframes))

                # obtain map points visible in the neighbouring key frames
                map_points_local = copy.deepcopy(map_points)
                print("Size of local map: {} - Size of global map: {}".format(
                    len(map_points_local.idx), len(map_points.idx)))
                delete_idxs = []
                for map_point_idx, observation in reversed(
                    list(enumerate(map_points.observations))):
                    if not any([keyframe_idx in neighbors_keyframes
                            for keyframe_idx in observation.keys()]):
                        delete_idxs.append(map_point_idx)
                        del map_points_local.observations[map_point_idx]
                if len(delete_idxs) > 0:
                    delete_idxs = np.hstack(delete_idxs)
                    map_points_local.idx = np.delete(
                        map_points_local.idx, delete_idxs, axis=0)
                    map_points_local.pts_3d = np.delete(
                        map_points_local.pts_3d, delete_idxs, axis=0)
                    map_points_local.representative_orb = np.delete(
                        map_points_local.representative_orb, delete_idxs, axis=0)

                print("Size of local map: {} - Size of global map: {}".format(len(map_points_local.idx), len(map_points.idx)))


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
                    kp = cv2.KeyPoint_convert(kp)
                    des = pose_graph.nodes[node_id]["des"]
                    #kp_matched = np.array(pose_graph.nodes[node_id]["kp_matched"])

                    # possible improvements:
                    # compute fundamental matrix from essential matrix
                    # use cv2.computeCorrespondEpilines to compute epipolar lines in this key frame (with node_id)
                    # search along epipolar line for matches instead of building the mask and brute force matching

                    # build a mask for ORB descriptor matching which permits only matches of nearby points
                    # this is slow: it would be better if we could ignore the keypoints that are already matched
                    max_distance = 8.0  # px
                    mask = np.zeros((len(map_points_local.representative_orb), len(des)), dtype=np.uint8)
                    kdtree = KDTree(kp.reshape(-1, 2).astype(np.uint16))  # KD-tree for fast lookup of neighbors
                    neighbor_kp_idxs = kdtree.query_ball_point(projected_pts.reshape(-1, 2), r=max_distance)
                    for map_point_idx, kp_idxs in enumerate(neighbor_kp_idxs):
                        for kp_idx in kp_idxs:
                            mask[map_point_idx, kp_idx] = 1

                    # find matches between projected map points and descriptors
                    distance_threshold = 20.0
                    bf_local = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # set to False
                    matches = bf_local.match(map_points_local.representative_orb, des, mask)  # TODO: select only those map points of the local group
                    # filter out matches with distance (descriptor appearance) greater than threshold
                    #matches = [m for m in matches if m.distance < distance_threshold]
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


                print(Counter(sorted([len([v for v in ob.values() if v is not None]) for ob in map_points.observations])))
                print(Counter(sorted([len(ob) for ob in map_points.observations])))



                # TODO:
                # For each map point
                # 1) project map point into each keyframe
                # 2) ignore keyframes in which the projection is out of the frame bounds
                # 3) search in a local region around the projected point for still
                #    unmatched ORB descriptors (using the representative ORB
                #    descriptor of the map point)
                # 4) If a match is found, add the corresponding keyframe to the list of observing keyframes for that map point
                # 5) Update the representative ORB descriptor of that map point

                # for node_id in sorted(pose_graph.nodes):
                #     # project map points into each key frame
                #     R, t = from_twist(pose_graph.nodes[node_id]["pose"])
                #     projected_pts, _ = cv2.projectPoints(map_points.pts_3d,
                #         R.T, -R.T.dot(t), camera_matrix, None)
                #     # filter out those which do not lie within the frame bounds
                #     #projected_pts, _ = get_visible_points(projected_pts,
                #     #    frame_width=frame.shape[1], frame_height=frame.shape[0])
                #
                #     # visualize projected map points
                #     if node_id == prev_node_id:
                #         for pts in projected_pts:
                #             match_frame = cv2.circle(match_frame, (int(pts[0, 0]), int(pts[0, 1])), 4, (0, 0, 255))
                #
                #     #pickle.dump(projected_pts, open("projected_pts_{}.pkl".format(node_id), "wb"))
                #
                # pose_graph_ = pose_graph.copy()
                # for node in pose_graph_.nodes:
                #     pose_graph_.nodes[node]["kp"] = cv2.KeyPoint_convert(pose_graph_.nodes[node]["kp"])
                # pickle.dump(pose_graph_, open("pose_graph_.pkl", "wb"))
                # pickle.dump(map_points, open("map_points_.pkl", "wb"))

                    # for each projected point visible in KF[node_id]
                    # search for matches with keypointsin local neighborhod of projected point
                    # if match could be found update the visible KF and associated kp indices of the corresponding map point

                    # for each projected map point retrieve keypoints in current frame which are in the local neighborhood of the projected point



                    # search neighborhood of each projected point for unmatched ORB
                    # descriptor that is similar to the representative descriptor
                    # of the map point


                # ###################################################################
                # #
                # # local bundle adjustment (over last 5 key frames)
                # #
                # ###################################################################
                #
                # if len(pose_graph.nodes) > 2:
                #
                #     # setup optimizer and camera parameters
                #     robust_kernel = True
                #     optimizer = g2o.SparseOptimizer()
                #     solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
                #     solver = g2o.OptimizationAlgorithmLevenberg(solver)
                #     optimizer.set_algorithm(solver)
                #
                #     focal_length = (camera_matrix[0,0] + camera_matrix[1,1]) / 2
                #     principal_point = (camera_matrix[0,2], camera_matrix[1,2])
                #     print("focal_length: ", focal_length, "principal_point: ", principal_point)
                #     cam = g2o.CameraParameters(focal_length, principal_point, 0)
                #     cam.set_id(0)
                #     optimizer.add_parameter(cam)
                #
                #     # add current keyframe poses
                #     true_poses = []
                #     nodes = list(sorted(pose_graph.nodes))[-5:]
                #     for i, node_id in enumerate(nodes):
                #         print("Using keyframe {} for local BA".format(node_id))
                #         R, t = from_twist(pose_graph.nodes[node_id]["pose"])
                #         pose = g2o.SE3Quat(R, np.squeeze(t))
                #         true_poses.append(pose)
                #
                #         v_se3 = g2o.VertexSE3Expmap()
                #         v_se3.set_id(node_id)
                #         v_se3.set_estimate(pose)
                #         #if i < 2:
                #         #if i == 0:
                #             #v_se3.set_fixed(True)
                #         optimizer.add_vertex(v_se3)
                #
                #     # add map points
                #     point_id = len(pose_graph) #len(true_poses)
                #     inliers = dict()
                #     for i, (point, observing_keyframes, associated_kp_indices) in enumerate(zip(map_points.pts_3d, map_points.observing_keyframes, map_points.associated_kp_indices)):
                #
                #         # skip points not visible in the selected subset of key frames
                #         if (observing_keyframes[0] not in nodes) or (observing_keyframes[1] not in nodes):
                #             continue
                #
                #         vp = g2o.VertexSBAPointXYZ()
                #         vp.set_id(point_id)
                #         vp.set_marginalized(True)
                #         vp.set_estimate(point)
                #         optimizer.add_vertex(vp)
                #
                #         # TODO:
                #         # 1) retrieve all key frames in which this map point is visible
                #         # 2) retrieve pixel coordinates of the keypoint corresponding to this map point in each key frame from 1)
                #         # 3) add an edge for each observation of the map point as below
                #         for node_id, kp_idx in zip(observing_keyframes, associated_kp_indices):
                #             #if node_id not in nodes:
                #             #    continue
                #             kp = cv2.KeyPoint_convert(pose_graph.nodes[node_id]["kp"])
                #             measurement = kp[kp_idx]
                #             #print(i, point_id, node_id, measurement)
                #
                #             edge = g2o.EdgeProjectXYZ2UV()
                #             edge.set_vertex(0, vp)  # map point
                #             edge.set_vertex(1, optimizer.vertex(node_id))  # pose of observing keyframe
                #             edge.set_measurement(measurement)   # needs to be set to the keypoint pixel position corresponding to that map point in that key frame (pose)
                #             edge.set_information(np.identity(2))
                #             if robust_kernel:
                #                 #edge.set_robust_kernel(g2o.RobustKernelHuber())
                #                 edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))  # 95% CI
                #
                #             edge.set_parameter_id(0, 0)
                #             optimizer.add_edge(edge)
                #
                #         inliers[point_id] = i
                #         point_id += 1
                #
                #     print('num vertices:', len(optimizer.vertices()))
                #     print('num edges:', len(optimizer.edges()))
                #
                # print('Performing full BA:')
                # optimizer.initialize_optimization()
                # optimizer.set_verbose(True)
                # optimizer.optimize(10)
                #
                # # # read out optimized poses
                # for node_id in nodes:
                #     print(node_id)
                #     vp = optimizer.vertex(node_id)
                #     se3quat = vp.estimate()
                #     R = np.copy(se3quat.to_homogeneous_matrix()[0:3, 0:3])
                #     t = np.copy(se3quat.to_homogeneous_matrix()[0:3, 3])
                #     pose_graph.nodes[node_id]["pose"] = to_twist(R, t)
                #
                # # read out optimized map points
                # for point_id, i in inliers.items():
                #     vp = optimizer.vertex(point_id)
                #     map_points.pts_3d[i, :] = np.copy(vp.estimate())


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
