import os
import csv
import pickle
from collections import defaultdict
import numpy as np

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


def triangulate_modules(tracks_file, patches_meta_dir, pose_graph, camera_matrix):
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
