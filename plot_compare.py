import sys
sys.path.append('/home/lukas/Pangolin/build/src')
sys.path.append('/home/pangolin/build/src') # for inside docker container

import os
import pickle
import random
import numpy as np
import cv2

import pypangolin as pango
from OpenGL.GL import *
from pytransform3d.rotations import axis_angle_from_matrix

from mapper.map_points import MapPoints
from mapper.geometry import from_twist
from mapper.modules import triangulate_modules


# load dumped map and camera trajectory
camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
pose_graph = pickle.load(open("pose_graph.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))
map_points = map_points.pts_3d

try:
    patches_meta_dir = os.path.join("data_processing", "patches", "meta")
    tracks_file = os.path.join("data_processing", "tracking", "tracks_cluster_000000.csv")
    module_corners, module_centers = triangulate_modules(tracks_file, patches_meta_dir, pose_graph, camera_matrix)
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
