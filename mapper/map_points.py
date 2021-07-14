import copy
import numpy as np


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

        representative_orb (`numpy.ndarray`): Each row is the representative
            ORB descriptor corresponding to the map point in pts_3d. This is an
            array of dtype uint8 and shape (num_map_points, 32). The most
            representative descriptor is the one which has the smallest Hamming
            distance to all other descriptors of that map points in the other
            observing key frames.
    """
    def __init__(self):
        """Map points"""
        self.idx = None
        self.pts_3d = np.empty(shape=(0, 3), dtype=np.float64)
        self.representative_orb = np.empty(shape=(0, 32), dtype=np.uint8)
        self.observing_keyframes = []
        self.associated_kp_indices = []

    def insert(self, new_map_points, associated_kp_indices,
            representative_orb, observing_kfs):
        """Add new map points into the exiting map."""
        if self.idx is not None:
            self.idx = np.hstack(
                (self.idx, np.arange(self.idx[-1]+1,
                self.idx[-1]+1+new_map_points.shape[0])))
        else:
            self.idx = np.arange(0, new_map_points.shape[0])
        self.pts_3d = np.vstack((self.pts_3d, new_map_points))
        for _ in range(new_map_points.shape[0]):
            self.observing_keyframes.append(copy.copy(observing_kfs))
        self.associated_kp_indices.extend(associated_kp_indices)
        self.representative_orb = np.vstack(
            (self.representative_orb, representative_orb))

    # def update_observing_keyframes(new_kfs, new_associated_kp_indices):

    # def update_representative_orb()

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

    #def update(self, keyframe_idx, new_pts_3d):
    #    """Update the map points of a keyframe. Needed e.g. for bundle adjustment."""
