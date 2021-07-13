###################################################################
#
# local bundle adjustment
#
###################################################################

## setup optimizer and camera parameters
robust_kernel = True
optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
solver = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(solver)

focal_length = (camera_matrix[0,0] + camera_matrix[1,1]) / 2
principal_point = (camera_matrix[0,2], camera_matrix[1,2])
print("focal_length: ", focal_length, "principal_point: ", principal_point)
cam = g2o.CameraParameters(focal_length, principal_point, 0)
cam.set_id(0)
optimizer.add_parameter(cam)

# add current keyframe poses
poses = []
for i, node_id in enumerate(sorted(pose_graph.nodes)):
    R, t = from_twist(pose_graph.nodes[node_id]["pose"])
    pose = g2o.SE3Quat(R, np.squeeze(t))
    # problems: (fixed)
    # my twist coordinates are first rotation, then translation
    # Se3Quat minimal vector is first translation, then rotation
    # rotation seems to be encoded differently in g2o minimal vector
    poses.append(pose)

    v_se3 = g2o.VertexSE3Expmap()
    print("pose i ", i)
    v_se3.set_id(i)
    v_se3.set_estimate(pose)
    if i < 2:
        v_se3.set_fixed(True)
    optimizer.add_vertex(v_se3)

# add map points
point_id = len(poses)
inliers = dict()
for i, point in enumerate(map_points.pts_3d):
    visible = []
    for j, pose in enumerate(poses):
        z = cam.cam_map(pose * point)
        if 0 <= z[0] < 640 and 0 <= z[1] < 512:
            visible.append((j, z))
    if len(visible) < 2:
        continue

    vp = g2o.VertexSBAPointXYZ()
    vp.set_id(point_id)
    vp.set_marginalized(True)
    vp.set_estimate(point)
    optimizer.add_vertex(vp)

    for j, z in visible:
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, vp)
        edge.set_vertex(1, optimizer.vertex(j))
        edge.set_measurement(z)
        edge.set_information(np.identity(2))
        if robust_kernel:
            #edge.set_robust_kernel(g2o.RobustKernelHuber())
            edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))  # 95% CI

        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)

    inliers[point_id] = i
    point_id += 1

print('num vertices:', len(optimizer.vertices()))
print('num edges:', len(optimizer.edges()))

print('Performing full BA:')
optimizer.initialize_optimization()
optimizer.set_verbose(True)
optimizer.optimize(10)

# # read out optimized poses
for i in range(len(poses)):
    vp = optimizer.vertex(i)
    se3quat = vp.estimate()
    R = np.copy(se3quat.to_homogeneous_matrix()[0:3, 0:3])
    t = np.copy(se3quat.to_homogeneous_matrix()[0:3, 3])
    pose_graph.nodes[i]["pose"] = to_twist(R, t)

# read out optimized map points
for i, point_id in enumerate(inliers):
    vp = optimizer.vertex(point_id)
    map_points.pts_3d[i, :] = np.copy(vp.estimate())
