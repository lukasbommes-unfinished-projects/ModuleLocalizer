import g2o
import cv2
import numpy as np

from mapper.geometry import from_twist, to_twist


def bundle_adjust(pose_graph, map_points, nodes, camera_matrix,
    keypoint_scale_levels, robust_kernel_value=None, verbose=True):

    inv_keypoint_scale_levels2 = 1.0/np.square(keypoint_scale_levels)

    # setup optimizer and camera parameters
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    focal_length = (camera_matrix[0,0] + camera_matrix[1,1]) / 2
    principal_point = (camera_matrix[0,2], camera_matrix[1,2])
    if verbose:
        print("focal_length: ", focal_length,
              "principal_point: ", principal_point)
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    # add poses of current keyframe and its direct neighbors
    for i, node_id in enumerate(sorted(nodes)):
        if verbose:
            print("Adding keyframe {} to local BA".format(node_id))
        R, t = from_twist(pose_graph.nodes[node_id]["pose"])
        pose = g2o.SE3Quat(R.T, np.squeeze(-R.T.dot(t)))
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(node_id)
        v_se3.set_estimate(pose)
        if i < 2:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)

    # add map points
    point_id = len(pose_graph)
    inliers = dict()
    for i, (point, observation) in enumerate(
            zip(map_points.pts_3d, map_points.observations)):

        # skip points not visible in the selected subset of key frames
        if not any([node_id in nodes for node_id in observation.keys()]):
            continue

        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(point)
        optimizer.add_vertex(vp)

        # 1) add an edge between the map point and all keyframes in
        #    which it is visible
        # 2) obtain the pixel coordinates of the keypoint associated
        #    to the map point in the respective key frame and add as
        #    measurement to the edge
        for node_id, kp_idx in observation.items():
            if node_id not in nodes:
                continue
            if kp_idx is None:
                continue
            kp = pose_graph.nodes[node_id]["kp"][kp_idx]

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)  # map point
            edge.set_vertex(1, optimizer.vertex(node_id))  # pose of observing keyframe
            edge.set_measurement(kp.pt)   # keypoint pixel position corresponding to that map point in that key frame
            keypoint_scale_levels
            inv_sigma2 = inv_keypoint_scale_levels2[kp.octave]
            edge.set_information(np.identity(2)*inv_sigma2)
            if robust_kernel_value:
                edge.set_robust_kernel(g2o.RobustKernelHuber(
                    robust_kernel_value))

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        inliers[point_id] = i
        point_id += 1

    if verbose:
        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))
        print('Performing full BA')

    optimizer.initialize_optimization()
    optimizer.set_verbose(verbose)
    optimizer.optimize(200)

    # # read out optimized poses
    for node_id in nodes:
        vp = optimizer.vertex(node_id)
        se3quat = vp.estimate()
        R = np.copy(se3quat.to_homogeneous_matrix()[0:3, 0:3])
        t = np.copy(se3quat.to_homogeneous_matrix()[0:3, 3])
        pose_graph.nodes[node_id]["pose"] = to_twist(R.T, -R.T.dot(t))

    # read out optimized map points
    for point_id, i in inliers.items():
        vp = optimizer.vertex(point_id)
        map_points.pts_3d[i, :] = np.copy(vp.estimate())
