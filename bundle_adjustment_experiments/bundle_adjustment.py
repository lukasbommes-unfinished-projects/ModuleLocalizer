import os
import pickle
import copy
import numpy as np
import g2o
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append("..")
from mapper.geometry import from_twist, to_twist
from mapper.modules import triangulate_modules

np.random.seed(0)


camera_matrix = pickle.load(open("../camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
pose_graph = pickle.load(open("pose_graph.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))

print(camera_matrix)

print(Counter(sorted([len([v for v in ob.values() if v is not None]) for ob in map_points.observations])))
print(Counter(sorted([len(ob) for ob in map_points.observations])))

map_points_bak = copy.deepcopy(map_points)
pose_graph_bak = pose_graph.copy()


###################################################################
#
# local bundle adjustment
#
###################################################################

newest_node_id = list(sorted(pose_graph.nodes))[-1]
print("Bundle adjustment for keyframe {}".format(newest_node_id))

# get node_ids of neighboring keyframes
neighbors_keyframes = [node_id for _, node_id in sorted(
    pose_graph.edges(newest_node_id))]
print("Neighboring keyframes: {}".format(neighbors_keyframes))
nodes = [*neighbors_keyframes, newest_node_id]

#nodes = list(sorted(pose_graph.nodes))

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

print("Poses before optimization")
for node_id in nodes:
    print(node_id)
    R, t = from_twist(pose_graph.nodes[node_id]["pose"])
    print(R, t)

# add current keyframe poses
true_poses = []
for i, node_id in enumerate(nodes):
    print("Using keyframe {} for local BA".format(node_id))
    R, t = from_twist(pose_graph.nodes[node_id]["pose"])
    t = -R.T.dot(t)
    R = R.T
    pose = g2o.SE3Quat(R, np.squeeze(t))
    true_poses.append(pose)

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(node_id)
    v_se3.set_estimate(pose)
    #if node_id == 0 or node_id == 1:
    if i < 2:
        v_se3.set_fixed(True)
    optimizer.add_vertex(v_se3)

# add map points
point_id = len(pose_graph)
inliers = dict()
for i, (point, observation) in enumerate(zip(map_points.pts_3d, map_points.observations)):

    # skip points not visible in the selected subset of key frames
    if not any([node_id in nodes for node_id in observation.keys()]):
        continue

    vp = g2o.VertexSBAPointXYZ()
    vp.set_id(point_id)
    vp.set_marginalized(True)
    vp.set_estimate(point)
    optimizer.add_vertex(vp)


    for node_id, kp_idx in observation.items():
        if node_id not in nodes:
            continue
        if kp_idx is None:
            continue
        kp = pose_graph.nodes[node_id]["kp"]
        #print("kp[kp_idx]", kp[kp_idx], kp[kp_idx].shape, kp[kp_idx].dtype, kp_idx, node_id)
        measurement = kp[kp_idx]
        #print(i, point_id, node_id, measurement)

        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, vp)  # map point
        edge.set_vertex(1, optimizer.vertex(node_id))  # pose of observing keyframe
        edge.set_measurement(measurement)   # needs to be set to the keypoint pixel position corresponding to that map point in that key frame (pose)
        edge.set_information(np.identity(2))
        if robust_kernel:
            edge.set_robust_kernel(g2o.RobustKernelHuber(1.96*np.std(map_points.pts_3d)))  # 95 % confidence interval of entire map, TODO: compute only from local map
            #edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))  # 5.991: 95% CI

        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)

    inliers[point_id] = i
    point_id += 1

print('num vertices:', len(optimizer.vertices()))
print('num edges:', len(optimizer.edges()))

print('Performing full BA:')
optimizer.initialize_optimization()
optimizer.set_verbose(True)
optimizer.optimize(200)

# read out optimized poses
print("Poses after optimization")
for node_id in nodes:
    print(node_id)
    vp = optimizer.vertex(node_id)
    se3quat = vp.estimate()
    R = np.copy(se3quat.to_homogeneous_matrix()[0:3, 0:3])
    t = np.copy(se3quat.to_homogeneous_matrix()[0:3, 3])
    t = -R.T.dot(t)
    R = R.T
    print(R, t)
    pose_graph.nodes[node_id]["pose"] = to_twist(R, t)  # why is that minus needed???

# read out optimized map points
for point_id, i in inliers.items():
    vp = optimizer.vertex(point_id)
    map_points.pts_3d[i, :] = np.copy(vp.estimate())

print("Map Points 3D L2 norm before vs. after optimization")
np.linalg.norm(map_points.pts_3d - map_points_bak.pts_3d, axis=0)

print("Camera parameters after optimization")
print(cam.focal_length, cam.principle_point, cam.baseline)








###########################   Visualization   ###########################
import sys
sys.path.append('/home/lukas/Pangolin/build/src')
sys.path.append('/home/pangolin/build/src') # for inside docker container

import pypangolin as pango
from OpenGL.GL import *
from pytransform3d.rotations import axis_angle_from_matrix


module_corners = {}
module_centers = {}
try:
    patches_meta_dir = os.path.join("..", "data_processing", "patches", "meta")
    tracks_file = os.path.join("..", "data_processing", "tracking", "tracks_cluster_000000.csv")
    module_corners, module_centers = triangulate_modules(tracks_file, patches_meta_dir, pose_graph, camera_matrix)
except FileNotFoundError:
    pass


def draw_camera_poses(poses, cam_scale, cam_aspect, color=(1.0, 0.6667, 0.0)):
    for R, t in [from_twist(pose) for pose in poses]:
        glPushMatrix()
        glTranslatef(*t)
        r = axis_angle_from_matrix(R) # returns x, y, z, angle
        r[-1] = r[-1]*180.0/np.pi  # rad -> deg
        glRotatef(r[3], r[0], r[1], r[2])  # angle, x, y, z
        glBegin(GL_LINES)
        glColor3f(1.0, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(cam_scale, 0, 0)

        glColor3f(0, 1.0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, cam_scale, 0)

        glColor3f(0, 0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, cam_scale)
        glEnd()

        glBegin(GL_LINE_LOOP)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(-1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(-1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glEnd()

        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(-1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(0, 0, 0)
        glVertex3f(1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(0, 0, 0)
        glVertex3f(1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(0, 0, 0)
        glVertex3f(-1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
        glVertex3f(0, 0, 0)
        glEnd()

        glPopMatrix()

    # connect camera poses with a line
    glLineWidth(3.0)
    glBegin(GL_LINE_STRIP)
    glColor3f(*color)
    for _, t in [from_twist(pose) for pose in poses]:
        glVertex3f(t[0, 0], t[1, 0], t[2, 0])
    glEnd()
    glLineWidth(1.0)


def draw_map_points(map_points, color=(0.5, 0.5, 0.5), size=2):
    glPointSize(size)
    glBegin(GL_POINTS)
    glColor3f(*color)
    for i, (p_x, p_y, p_z) in enumerate(zip(map_points[:, 0], map_points[:, 1], map_points[:, 2])):
        glVertex3f(p_x, p_y, p_z)
    glEnd()


def plot():
    win = pango.CreateWindowAndBind("pySimpleDisplay", 1600, 900)
    glEnable(GL_DEPTH_TEST)

    aspect = 1600/900
    cam_scale = 0.5
    cam_aspect = aspect

    pm = pango.ProjectionMatrix(1600,900,1000,1000,800,450,0.1,1000);
    mv = pango.ModelViewLookAt(0, 0, -1, 0, 0, 0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)

    handler=pango.Handler3D(s_cam)
    d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
                                            pango.Attach(1),
                                            pango.Attach(0),
                                            pango.Attach(1),
                                            -aspect).SetHandler(handler)

    while not pango.ShouldQuit():
        #glClearColor(0.0, 0.5, 0.0, 1.0)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        glMatrixMode(GL_MODELVIEW)

        # draw map points
        draw_map_points(map_points_bak.pts_3d, color=(0.5, 0.5, 0.5), size=4)
        draw_map_points(map_points.pts_3d, color=(0.0, 0.0, 1.0), size=4)

        # draw origin coordinate system (red: x, green: y, blue: z)
        glBegin(GL_LINES)
        glColor3f(1.0, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(3, 0, 0)

        glColor3f(0, 1.0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 3, 0)

        glColor3f(0, 0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 3)
        glEnd()

        # draw every camera pose as view frustrum
        poses = [pose_graph_bak.nodes[n]["pose"] for n in pose_graph_bak]
        draw_camera_poses(poses, cam_scale, cam_aspect, color=(1.0, 0.6667, 0.0))

        poses = [pose_graph.nodes[n]["pose"] for n in pose_graph]
        draw_camera_poses(poses, cam_scale, cam_aspect, color=(0.0, 0.6667, 1.0))

        # draw PV modules + corners
        for track_id, points in module_corners.items():
            glPointSize(10)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            for p_x, p_y, p_z in zip(points[:, 0], points[:, 1], points[:, 2]):
                glVertex3f(p_x, p_y, p_z)
            glEnd()

            glBegin(GL_LINE_LOOP)
            for p_x, p_y, p_z in zip(points[:, 0], points[:, 1], points[:, 2]):
                glVertex3f(p_x, p_y, p_z)
            glEnd()

        # draw PV module centers
        for track_id, points in module_centers.items():
            glPointSize(10)
            glBegin(GL_POINTS)
            glColor3f(0.0, 1.0, 0.0)
            for p_x, p_y, p_z in zip(points[:, 0], points[:, 1], points[:, 2]):
                glVertex3f(p_x, p_y, p_z)
            glEnd()

        pango.FinishFrame()

plot()
