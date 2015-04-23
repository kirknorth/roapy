"""
grid.util.transform
===================

"""

import numpy as np
import pyproj


def standard_refraction(radar, debug=False, verbose=False):
    """
    Transform radar polar coordinates to Cartesian coordinates assuming
    standard atmospheric refraction, the so-called 4/3 Earth's radius rule.
    """

    # Effective radius of Earth
    Re = 6371.0 * 4.0 / 3.0 * 1000.0

    # Parse radar scan geometry
    # Convert scan angles to radians
    rng = radar.range['data']
    ele = np.radians(radar.elevation['data'])
    azi = np.radians(radar.azimuth['data'])

    # Create radar coordinates mesh
    ELE, RNG = np.meshgrid(ele, rng, indexing='ij')
    AZI, RNG = np.meshgrid(azi, rng, indexing='ij')

    # Flatten radar coordinates
    RNG, AZI, ELE = RNG.flatten(), AZI.flatten(), ELE.flatten()

    # Compute vertical height (z), arc length (s), eastward distance (x),
    # and northward distance (y)
    z = np.sqrt(RNG**2 + 2.0 * RNG * Re * np.sin(ELE) + Re**2) - Re
    s = Re * np.arcsin(RNG * np.cos(ELE) / (z + Re))
    x = s * np.sin(AZI)
    y = s * np.cos(AZI)

    return z, y, x


def _calculate_radar_offset(
        radar, lat_0=None, lon_0=None, alt_0=None, proj='lcc', datum='NAD83',
        ellps='GRS80', debug=False, verbose=False):
    """
    Calculate the (x, y, z) Cartesian coordinates of a radar within the an
    analysis domain centered at the specified latitude and longitude.

    Parameters
    ----------
    radar : Radar
        Radar object with altitude, latitude, and longitude information.
    lat_0, lon_0, alt_0 : float
        The latitude, longitude, and altitude AMSL of the grid origin,
        respectively. The default uses the location of the radar as the grid
        origin.
    proj : str
        See pyproj documentation for more information.
    datum : str
        See pyproj documentation for more information.
    ellps : str
        See pyproj documentation for more information.
    debug : bool
        True to print debugging information, False to suppress.
    verbose : bool
        True to print progress information, False to suppress.

    Return
    ------
    offset : tuple
        The radar (z, y, x) Cartesian location within the analysis domain.
    """

    # TODO: revisit altitude information for consistency, e.g., AGL versus AMSL

    # Parse radar location data
    radar_lat = radar.latitude['data'][0]
    radar_lon = radar.longitude['data'][0]
    radar_alt = radar.altitude['data'][0]

    # Parse analysis domain origin location data
    if lat_0 is None:
        lat_0 = radar_lon
    if lon_0 is None:
        lon_0 = radar_lon
    if alt_0 is None:
        alt_0 = radar_alt

    # Create map projection
    pj = pyproj.Proj(
        proj=proj, lat_0=lat_0, lon_0=lon_0, x_0=0.0, y_0=0.0, datum=datum,
        ellps=ellps)

    # Compute the radar (z, y, x) coordinates defined by map projection
    radar_x, radar_y = pj(radar_lon, radar_lat)
    radar_z = radar_alt - alt_0

    if debug:
        print 'Radar z offset from origin: {:.2f} km'.format(radar_z / 1000.0)
        print 'Radar y offset from origin: {:.2f} km'.format(radar_y / 1000.0)
        print 'Radar x offset from origin: {:.2f} km'.format(radar_x / 1000.0)

    return radar_z, radar_y, radar_x
