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


def from_twist(twist):
    """Convert a 6D twist coordinate (shape (6,)) into a 3x3 rotation matrix
    and translation vector (shape (3,))."""
    r = twist[:3].reshape(3, 1)
    t = twist[3:].reshape(3, 1)
    R, _ = cv2.Rodrigues(r)
    return R, t


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


def sort_cw(pts):
    """Sort points clockwise by first splitting
    left/right points and then top/bottom."""
    pts = [list(p) for p in pts.reshape(-1, 2)]
    pts_sorted = sorted(pts , key=lambda k: k[0])
    pts_left = pts_sorted[:2]
    pts_right = pts_sorted[2:]
    pts_left_sorted = sorted(pts_left , key=lambda k: k[1])
    pts_right_sorted = sorted(pts_right , key=lambda k: k[1])
    tl = pts_left_sorted[0]
    bl = pts_left_sorted[1]
    tr = pts_right_sorted[0]
    br = pts_right_sorted[1]
    return [tl, tr, br, bl]


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
        tracks_first_filtered: (`dict`) Track IDs and mask names of all tracked modules which
            occur in both the first and second frame. Contains the mask names as they occur in the
            first frame.

        tracks_second_filtered: (`dict`) Track IDs and mask names of all tracked modules which
            occur in both the first and second frame. Contains the mask names as they occur in the
            second frame.
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
        # sort corners of quadrilateral
        tl, tr, br, bl = sort_cw(meta["quadrilateral"].reshape(-1, 2))
        quadrilateral_sorted = np.array([[tl],[tr],[br],[bl]])
        quadrilaterals.append(quadrilateral_sorted.astype(np.float64))
    return centers, quadrilaterals

# load dumped map and camera trajectory
kf_poses = pickle.load(open("kf_poses.pkl", "rb"))
kf_frame_names = pickle.load(open("kf_frame_names.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))
map_points = map_points.pts_3d


# triangulate module centers and corners
patches_meta_dir = os.path.join("data_processing", "patches", "meta")
tracks_file = os.path.join("data_processing", "tracking", "tracks_cluster_000000.csv")
tracks = load_tracks(tracks_file)
camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
module_corners = defaultdict(list)
module_centers = defaultdict(list)

for first_pose, second_pose, first_frame_name, second_frame_name in zip(
        kf_poses, kf_poses[1:], kf_frame_names, kf_frame_names[1:]):

    tracks_first_filtered, tracks_second_filtered = get_common_modules(
        tracks, first_frame_name, second_frame_name)

    centers_first, quadrilaterals_first = get_module_corners(
        patches_meta_dir, first_frame_name, tracks_first_filtered.values())

    centers_second, quadrilaterals_second = get_module_corners(
        patches_meta_dir, second_frame_name, tracks_second_filtered.values())

    # triangulate 3D points
    R1, t1 = from_twist(first_pose)
    R2, t2 = from_twist(second_pose)

    for i, track_id in enumerate(tracks_first_filtered.keys()):
        last_pts = quadrilaterals_first[i]
        current_pts = quadrilaterals_second[i]
        module_corners[track_id].append(triangulate_map_points(
            last_pts, current_pts, R1, t1, R2, t2, camera_matrix))

        last_pts = centers_first[i]
        current_pts = centers_second[i]
        module_centers[track_id].append(triangulate_map_points(
            last_pts, current_pts, R1, t1, R2, t2, camera_matrix))

# compute medians of observed center points
module_corners = {track_id: np.median(np.stack(points), axis=0) for track_id, points in module_corners.items()}
module_centers = {track_id: np.median(np.stack(points), axis=0) for track_id, points in module_centers.items()}
#module_centers = {track_id: np.vstack(points) for track_id, points in module_centers.items()}
#module_corners = {track_id: np.vstack(points) for track_id, points in module_corners.items()}

COLORS = [(0, 0, 1),
          (0, 1, 0),
          (0, 1, 1),
          (1, 0, 0),
          (1, 0, 1),
          (1, 1, 0),
          (1, 1, 1)]

# extract keyframe poses
Rs = []
ts = []
for kf_pose in kf_poses:
    R, t = from_twist(kf_pose)
    Rs.append(R)
    ts.append(t)

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
        for R, t in zip(Rs, ts):
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
        for t in ts:
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
