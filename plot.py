import sys
sys.path.append('/home/lukas/Pangolin/build/src')
sys.path.append('/home/pangolin/build/src') # for inside docker container

import os
import csv
import pickle
import random
from collections import defaultdict
import numpy as np
import cv2

import pypangolin as pango
from OpenGL.GL import *
from pytransform3d.rotations import axis_angle_from_matrix

from mapper.map_points import MapPoints
from mapper.geometry import from_twist, triangulate_map_points


def load_tracks(tracks_file):
    """Load Tracks CSV file."""
    tracks = defaultdict(list)
    with open(tracks_file, newline='', encoding="utf-8-sig") as csvfile:  # specifying the encoding skips optional BOM
        # automatically infer CSV file format
        dialect = csv.Sniffer().sniff(csvfile.readline(), delimiters=",;")
        csvfile.seek(0)
        csvreader = csv.reader(csvfile, dialect)
        for row in csvreader:
            frame_name = row[0]
            tracks[frame_name].append((row[1], row[2]))  # mask_name, track_id
    return tracks


def get_common_modules(tracks, first_frame_name, second_frame_name):
    """Retrieve mask names and track IDs of modules visible in two different frames.

    Args:
        tracks (`dict` of `list` of `tuple`): Tracks CSV file as created by tracking step of
            PV Module Extractor. Each row has format frame_name, mask_name, track_id, center_x, center_y

        first_frame_name (`str`): Name of the first frame in which to look for tracked modules.

        second_frame_name (`str`): Name of the second frame in which to look for the same
            tracked modules as in first frame.

    Returns:
        tracks_first_filtered: (`dict`) Track IDs (keys) and mask names (values)
            of all tracked modules which occur in both the first and second
            frame. Contains the mask names as they occur in the first frame.

        tracks_second_filtered: (`dict`) Track IDs (keys) and mask names (values)
            of all tracked modules which occur in both the first and second
            frame. Contains the mask names as they occur in the second frame.
    """
    tracks_first = {t[1]: t[0] for t in tracks[first_frame_name]}
    tracks_second = {t[1]: t[0] for t in tracks[second_frame_name]}
    common_track_ids = set(tracks_first.keys()) & set(tracks_second.keys())
    tracks_first_filtered = {k: v for k, v in tracks_first.items() if k in common_track_ids}
    tracks_second_filtered = {k: v for k, v in tracks_second.items() if k in common_track_ids}
    return tracks_first_filtered, tracks_second_filtered


def get_module_corners(root_dir, frame_name, mask_names):
    """Retrieve corners and center points for a given mask (module) and frame."""
    centers = []
    quadrilaterals = []
    for mask_name in mask_names:
        meta_file = os.path.join(root_dir, frame_name, "{}.pkl".format(mask_name))
        meta = pickle.load(open(meta_file, "rb"))
        centers.append(np.array(meta["center"]).reshape(1, 1, 2).astype(np.float64))
        quadrilaterals.append(np.array(meta["quadrilateral"]).reshape(-1, 2).astype(np.float64))
    return centers, quadrilaterals


def triangulate_module_corners(tracks_file, patches_meta_dir, pose_graph, camera_matrix):
    """Triangulate 3D map points of PV module corners and centers."""
    tracks = load_tracks(tracks_file)
    module_corners = defaultdict(list)
    module_centers = defaultdict(list)

    kf_poses = [pose_graph.nodes[n]["pose"] for n in pose_graph]
    kf_frame_names = [pose_graph.nodes[n]["frame_name"] for n in pose_graph]

    for first_pose, second_pose, first_frame_name, second_frame_name in zip(
            kf_poses, kf_poses[1:], kf_frame_names, kf_frame_names[1:]):

        tracks_first_filtered, tracks_second_filtered = get_common_modules(
            tracks, first_frame_name, second_frame_name)
        centers_first, corners_first = get_module_corners(
            patches_meta_dir, first_frame_name,
            tracks_first_filtered.values())
        centers_second, corners_second = get_module_corners(
            patches_meta_dir, second_frame_name,
            tracks_second_filtered.values())

        # triangulate 3D points
        R1, t1 = from_twist(first_pose)
        R2, t2 = from_twist(second_pose)

        for i, track_id in enumerate(tracks_first_filtered.keys()):
            module_corners[track_id].append(triangulate_map_points(
                corners_first[i], corners_second[i],
                R1, t1, R2, t2, camera_matrix))
            module_centers[track_id].append(triangulate_map_points(
                centers_first[i], centers_second[i],
                R1, t1, R2, t2, camera_matrix))

    # compute medians of observed center points
    module_corners = {
        track_id: np.median(np.stack(points), axis=0)
        for track_id, points in module_corners.items()}
    module_centers = {
        track_id: np.median(np.stack(points), axis=0)
        for track_id, points in module_centers.items()}

    # do not comput emedians and return all points instead
    #module_centers = {track_id: np.vstack(points) for track_id, points in module_centers.items()}
    #module_corners = {track_id: np.vstack(points) for track_id, points in module_corners.items()}
    return module_corners, module_centers


# load dumped map and camera trajectory
camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
pose_graph = pickle.load(open("pose_graph.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))
map_points = map_points.pts_3d

try:
    patches_meta_dir = os.path.join("data_processing", "patches", "meta")
    tracks_file = os.path.join("data_processing", "tracking", "tracks_cluster_000000.csv")
    module_corners, module_centers = triangulate_module_corners(tracks_file, patches_meta_dir, pose_graph, camera_matrix)
except FileNotFoundError:
    module_corners = {}
    module_centers = {}

# COLORS = [(0, 0, 1),
#           (0, 1, 0),
#           (0, 1, 1),
#           (1, 0, 0),
#           (1, 0, 1),
#           (1, 1, 0),
#           (1, 1, 1)]

# print only every x map points
subsmaple_map_points = 1

def main():
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

        # draw map points (old format)
        glPointSize(2)
        glBegin(GL_POINTS)
        #for i, m in enumerate(map_points.pts_3d):
        # get_random_color
        #glColor3f(*COLORS[i%len(COLORS)])
        glColor3f(0.5, 0.5, 0.5)
        for i, (p_x, p_y, p_z) in enumerate(zip(map_points[:, 0], map_points[:, 1], map_points[:, 2])):
            if i % subsmaple_map_points != 0:
                continue
            glVertex3f(p_x, p_y, p_z)
        glEnd()


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
        for R, t in [from_twist(pose_graph.nodes[n]["pose"]) for n in pose_graph]:
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
        glColor3f(1.0, 0.6667, 0.0)
        for _, t in [from_twist(pose_graph.nodes[n]["pose"]) for n in pose_graph]:
            glVertex3f(t[0, 0], t[1, 0], t[2, 0])
        glEnd()
        glLineWidth(1.0)


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


        # TODO:
        # draw lines between cameras
        # visualize ground plane
        # project images onto ground plane
        # visualize local group

        pango.FinishFrame()


if __name__ == "__main__":
    main()
