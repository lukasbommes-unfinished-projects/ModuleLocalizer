import os
import cv2
import numpy as np


def to_celsius(image):
    """Convert raw intensity values of radiometric image to Celsius scale."""
    return image*0.04-273.15


def preprocess_radiometric_frame(frame, equalize_hist=True):
    """Preprocesses raw radiometric frame.

    First, the raw 16-bit radiometric intensity values are converted to Celsius
    scale. Then, the image values are normalized to range [0, 255] and converted
    to 8-bit. Finally, histogram equalization is performed to normalize
    brightness and enhance contrast.
    """
    frame = to_celsius(frame)
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    frame = (frame*255.0).astype(np.uint8)
    if equalize_hist:
        frame = cv2.equalizeHist(frame)
        # CLAHE results in vastly different numbers of feature points depending on clipLimit
        #clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
        #frame = clahe.apply(frame)
    return frame


class Capture:
    def __init__(self, image_files, mask_files=None, camera_matrix=None,
            dist_coeffs=None):
        self.frame_counter = 0
        self.image_files = image_files
        self.mask_files = mask_files
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.num_images = len(self.image_files)
        # precompute undistortion maps
        probe_frame = cv2.imread(self.image_files[0], cv2.IMREAD_ANYDEPTH)
        self.img_w = probe_frame.shape[1]
        self.img_h = probe_frame.shape[0]
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            #new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            #    self.camera_matrix, self.dist_coeffs,
            #    (self.img_w, self.img_h), alpha=1.0)
            new_camera_matrix = self.camera_matrix
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                new_camera_matrix, (self.img_w, self.img_h), cv2.CV_32FC1)
        if mask_files is not None:
            assert len(mask_files) == len(image_files), "Number of mask_files and image_files do not match"
            self.mask_files = mask_files


    def get_next_frame(self, preprocess=True, undistort=False,
            equalize_hist=True):
        frame = None
        masks = None
        frame_name = None
        mask_names = None
        if self.frame_counter < self.num_images:
            image_file = self.image_files[self.frame_counter]
            frame_name = str.split(os.path.basename(image_file), ".")[0]
            frame = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
            if self.mask_files is not None:
                mask_file = self.mask_files[self.frame_counter]
                masks = [cv2.imread(m, cv2.IMREAD_ANYDEPTH) for m in mask_file]
                mask_names = [str.split(os.path.basename(m), ".")[0] for m in mask_file]
            self.frame_counter += 1
            if preprocess:
                frame = preprocess_radiometric_frame(frame, equalize_hist)
            if undistort and self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_CUBIC)
                if self.mask_files is not None:
                    masks = [cv2.remap(mask, self.mapx, self.mapy, cv2.INTER_CUBIC) for mask in masks]
        return frame, masks, frame_name, mask_names


def get_visible_points(pts, frame_width, frame_height):
    """Remove points from pts which fall outside the frame."""
    min_bb = np.array([0, 0])
    max_bb = np.array([frame_width, frame_height])
    visible = np.all(np.logical_and(
        min_bb <= pts.reshape(-1, 2), pts.reshape(-1, 2) <= max_bb), axis=1)
    return pts[visible]


def update_matched_flag(pose_graph, matches):
    """Sets a boolean flag for each keypoint in the pose graph
    indicating whether this keypoint was already matched to
    another keypoint in another key frame.
    """
    train_idxs = [m.trainIdx for m in matches]
    query_idxs = [m.queryIdx for m in matches]
    prev_node_id = sorted(pose_graph.nodes)[-1]

    num_kps = len(pose_graph.nodes[prev_node_id]["kp"])
    pose_graph.nodes[prev_node_id]["kp_matched"] = [
        True if i in train_idxs else False for i in range(num_kps)]

    pose_graph.nodes[prev_node_id-1]
    num_kps = len(pose_graph.nodes[prev_node_id-1]["kp"])
    pose_graph.nodes[prev_node_id-1]["kp_matched"] = [
        True if i in query_idxs else False for i in range(num_kps)]
