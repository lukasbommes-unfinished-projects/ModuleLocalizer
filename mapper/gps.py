import numpy as np

from mapper.geotransforms import geodetic2enu


def interpolate_gps(gps):
    """Interpolates a piecewise constant gps signal with piecewise
    linear segments.

    This is needed because the GPS position acquiredby our drone is
    constant for several frames and then jumps abruptly. This
    function smooths out the abrupt changes by linearly interpolating
    positions between the jumps.
    """

    # compute differences at steps, i.e. at the points where the GPS signal updates
    diffs = np.diff(gps, axis=0)

    # find indices where jumps occur
    update_idxs = np.where(np.abs(diffs[:, 0]) > 0)[0]
    update_idxs += 1
    update_idxs = np.insert(update_idxs, 0, 0)
    update_idxs = np.insert(update_idxs, len(update_idxs), len(gps)-1)

    # compute position incrementals to smoothen out jumps
    gps_diffs = []
    for i, (last_update_idx, update_idx) in enumerate(zip(update_idxs, update_idxs[1:])):
        gps_diff = gps[update_idx] - gps[last_update_idx]
        n_steps = update_idx - last_update_idx
        for j in range(n_steps):
            gps_diffs.append(j*(gps_diff / n_steps))
    gps_diffs = np.array(gps_diffs)

    return gps[:-1, :] + gps_diffs


def gps_to_ltp(gps):
    """Converts GPS readings from WGS-84 (lat, lon, height) to local tangent plane.
    The first gps reading is choosen as origin.

    Args:
        gps (`numpy.ndarray`): Shape (-1, 3). Each row is a GPS position in
            WGS-84 coordinates of the form longitude (degrees), latitude
            (degrees), height (meters).

    Returns:
        gps_ltp (`numpy.ndarray`): Shape (-1, 3). Corresponding GPS position
            in local tangent plane coordinates East (meters), North (meters),
            height (meters). The origin of the local tangent plane is the first
            input gps position.

        origin (`tuple` of `float`): WGS-84 latitue, longitude and height of
            the selected origin of the local tangent plane.
    """
    lon0, lat0, h0 = gps[0, :]
    print(("Origin of local tangent plane: lat: {} deg -- long: {} deg "
          "-- height: {} m").format(lat0, lon0, h0))
    gps_ltp = np.zeros_like(gps)
    for i, (lon, lat, h) in enumerate(gps):
        e, n, u = geodetic2enu(lat, lon, h, lat0, lon0, h0)
        gps_ltp[i, :] = np.array([e, n, u])
    origin = (lat0, lon0, h0)
    return gps_ltp, origin
