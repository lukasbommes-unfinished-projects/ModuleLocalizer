import sys
sys.path.append('/home/lukas/Pangolin/build/src')
sys.path.append('/home/pangolin/build/src') # for inside docker container

import pypangolin as pango
from OpenGL.GL import *

import pickle
import random
import numpy as np
import cv2

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

#Rs = pickle.load(open("Rs.pkl", "rb"))
#ts = pickle.load(open("ts.pkl", "rb"))
kf_poses = pickle.load(open("kf_poses.pkl", "rb"))
#kf_visible_map_points = pickle.load(open("kf_visible_map_points.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))

map_points = map_points.pts_3d

COLORS = [(0, 0, 1),
          (0, 1, 0),
          (0, 1, 1),
          (1, 0, 0),
          (1, 0, 1),
          (1, 1, 0),
          (1, 1, 1)]

#COLORS = [(0.5, 0.5, 0.5)]

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


        # TODO:
        # draw lines between cameras
        # visualize ground plane
        # project images onto ground plane
        # visualize local group

        pango.FinishFrame()


if __name__ == "__main__":
    main()
