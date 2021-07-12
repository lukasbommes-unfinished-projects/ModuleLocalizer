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
kf_visible_map_points = pickle.load(open("kf_visible_map_points.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))

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

        # # draw map points (old format)
        # glPointSize(2)
        # glBegin(GL_POINTS)
        # for i, m in enumerate(map_points):
        #     # get_random_color
        #     glColor3f(*COLORS[i%len(COLORS)])
        #     for j, (p_x, p_y, p_z) in enumerate(zip(m[:, 0], m[:, 1], m[:, 2])):
        #         if j % subsmaple_map_points != 0:
        #             continue
        #         glVertex3f(p_x, p_y, p_z)
        # glEnd()

        # draw map points (new format)
        glPointSize(2)
        glBegin(GL_POINTS)
        for i, kfs in enumerate(kf_visible_map_points):
            # get_random_color
            glColor3f(*COLORS[i%len(COLORS)])
            visible_map_points = np.vstack([map_points[k, :] for k in kfs])
            for j, (p_x, p_y, p_z) in enumerate(zip(visible_map_points[:, 0], visible_map_points[:, 1], visible_map_points[:, 2])):
                if j % subsmaple_map_points != 0:
                    continue
                glVertex3f(p_x, p_y, p_z)
        glEnd()

        # #draw map points visible in two key frames
        # glPointSize(2)
        # glBegin(GL_POINTS)
        # glColor3f(1.0, 0.667, 0.0)
        # visible_map_points = np.vstack([map_points[k, :] for k in kf_visible_map_points[0]])  #5
        # for j, (p_x, p_y, p_z) in enumerate(zip(visible_map_points[:, 0], visible_map_points[:, 1], visible_map_points[:, 2])):
        #     #if j % subsmaple_map_points != 0:
        #     #    continue
        #     glVertex3f(p_x, p_y, p_z)
        #
        # glColor3f(0.0, 0.0, 1.0)
        # visible_map_points = np.vstack([map_points[k, :] for k in kf_visible_map_points[1]])  #33
        # for j, (p_x, p_y, p_z) in enumerate(zip(visible_map_points[:, 0], visible_map_points[:, 1], visible_map_points[:, 2])):
        #     #if j % subsmaple_map_points != 0:
        #     #    continue
        #     glVertex3f(p_x, p_y, p_z)
        # glEnd()

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
