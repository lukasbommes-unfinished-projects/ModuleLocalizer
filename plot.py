import sys
sys.path.append('/home/lukas/Pangolin/build/src')
sys.path.append('/home/pangolin/build/src') # for inside docker container

import os
import json
import pickle
import random
import numpy as np
import cv2

import pypangolin as pango
from OpenGL.GL import *
from pytransform3d.rotations import axis_angle_from_matrix

from mapper.map_points import MapPoints
from mapper.geometry import from_twist, transform_to_gps_frame, gps_to_ltp
from mapper.modules import triangulate_modules


# load dumped map and camera trajectory
camera_matrix = pickle.load(open("camera_calibration/parameters/ir/camera_matrix.pkl", "rb"))
pose_graph = pickle.load(open("pose_graph.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))

gps_file = "data_processing/splitted/gps/gps.json"
gps_ = json.load(open(gps_file, "r"))
gps = np.zeros((len(gps_), 3))
gps[:, 0:2] = np.array(gps_)
gps, _ = gps_to_ltp(gps)

gps_positions = transform_to_gps_frame(pose_graph, map_points, gps)

#keyframe_idxs =  [int(pose_graph.nodes[node_id]["frame_name"][6:]) for node_id in sorted(pose_graph.nodes)]
#gps_positions = np.array([gps[idx] for idx in keyframe_idxs])  # plot only gps track of keyframes
#gps_positions = np.array(gps)  # plot entire available gps track
#gps_positions = (gps_positions - gps_positions[0])*1/1.5312240158658566e-05 # TODO: compute with sim3 solver (in odometry.py)

# whether to visualize tracked PV modules
plot_modules = True
module_corners = {}
module_centers = {}
if plot_modules:
	try:
		patches_meta_dir = os.path.join("data_processing", "patches", "meta")
		tracks_file = os.path.join("data_processing", "tracking", "tracks_cluster_000000.csv")
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


def draw_map_points(map_points, color=(0.5, 0.5, 0.5), size=2, subsample=1):
	glPointSize(size)
	glBegin(GL_POINTS)
	glColor3f(*color)
	for i, (p_x, p_y, p_z) in enumerate(zip(map_points[:, 0], map_points[:, 1], map_points[:, 2])):
		if i % subsample != 0:
			continue
		glVertex3f(p_x, p_y, p_z)
	glEnd()


def draw_gps_track(gps_positions, size=10.0, color=(1.0, 0.0, 0.0)):
	glLineWidth(3.0)
	glBegin(GL_LINE_STRIP)
	glColor3f(*color)
	for t in gps_positions:
		glVertex3f(t[0], t[1], t[2])
	glEnd()
	glLineWidth(1.0)

	glPointSize(size)
	glBegin(GL_POINTS)
	glColor3f(*color)
	for t in gps_positions:
		glVertex3f(t[0], t[1], t[2])
	glEnd()


def draw_pv_modules(module_corners, module_centers):
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


def draw_origin():
	"""Draw origin coordinate system (red: x, green: y, blue: z)"""
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


def draw_ground_plane(ground_plane_z):
	glDisable(GL_TEXTURE_2D)
	glBegin(GL_QUADS)
	glNormal3f(0.0, 1.0, 0.0)
	z0 = ground_plane_z
	repeat = 20
	for y in range(repeat):
		yStart = 100.0 - y*10.0
		for x in range(repeat):
			xStart = x*10.0 - 100.0
			if ((y % 2) ^ (x % 2)):
				glColor4ub(41, 41, 41, 255)
			else:
				glColor4ub(200, 200, 200, 255)
			glVertex3f(xStart, yStart, z0)
			glVertex3f(xStart + 10.0, yStart, z0)
			glVertex3f(xStart + 10.0, yStart - 10.0, z0)
			glVertex3f(xStart, yStart - 10.0, z0)
	glEnd()


def plot():
	win = pango.CreateWindowAndBind("pySimpleDisplay", 1600, 900)
	glEnable(GL_DEPTH_TEST)

	aspect = 1600/900
	cam_scale = 0.5
	cam_aspect = aspect

	pm = pango.ProjectionMatrix(
		1600,  # width
		900,  # height
		1000,  # fu
		1000,  # fv
		800,  # u0
		450,  # v0
		0.1,  # z near
		1000)  # z far
	mv = pango.ModelViewLookAt(0, 0, -1, 0, 0, 0, pango.AxisY)
	s_cam = pango.OpenGlRenderState(pm, mv)

	handler = pango.Handler3D(s_cam)
	d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
											pango.Attach(1),
											pango.Attach(0),
											pango.Attach(1),
											-aspect).SetHandler(handler)

	# find z position of ground plane
	ground_plane_z = np.quantile(map_points.pts_3d[:, -1], 0.95) + 0.5

	while not pango.ShouldQuit():
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		d_cam.Activate(s_cam)
		glMatrixMode(GL_MODELVIEW)

		#draw_ground_plane(ground_plane_z)
		draw_origin()
		draw_map_points(map_points.pts_3d, color=(0.5, 0.5, 0.5), size=4)
		poses = [pose_graph.nodes[n]["pose"] for n in pose_graph]
		draw_camera_poses(poses, cam_scale, cam_aspect, color=(0.0, 0.6667, 1.0))
		draw_gps_track(gps_positions)
		draw_pv_modules(module_corners, module_centers)
		pango.FinishFrame()


if __name__ == "__main__":
	plot()
