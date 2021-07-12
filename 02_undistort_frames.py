import os
import glob
import cv2
from tqdm import tqdm

from common import Capture


def run(frames_root, output_dir, cam_params):

    for dirname in ["radiometric", "preview"]:
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)

    frame_files_radiometric = sorted(
        glob.glob(os.path.join(frames_root, "radiometric", "*")))
    frame_files_preview = sorted(
        glob.glob(os.path.join(frames_root, "preview", "*")))

    # probe frame width and height
    probe_frame = cv2.imread(frame_files_radiometric[0], cv2.IMREAD_ANYDEPTH)
    frame_width, frame_height = probe_frame.shape[1:3][::-1]
    print(frame_width, frame_height)

    # load camera parameters from pickle files
    camera_matrix = pickle.load(open("camera_matrix.pkl", "rb"))
    dist_coeffs = pickle.load(open("dist_coeffs.pkl", "rb"))

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (frame_width, frame_height), alpha=1.0)

    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3),
        new_camera_matrix, (width, height), cv2.CV_16SC2)

    for frame_file_radiometric, frame_file_preview in zip(
            frame_files_radiometric, frame_files_preview):
        cv2.imread(frame_file_radiometric, cv2.IMREAD_ANYDEPTH)




if __name__ == "__main__":
    frames_root = "workdir/splitted"
    output_dir = "workdir/undistorted"
    cam_params = "camera_calibration"
    run(frames_root, output_dir, cam_params)
