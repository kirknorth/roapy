"""
grid.util.transform
===================

"""

import numpy as np
import pyproj

# Necessary and/or potential improvements to the transform module:
#
# * Add the ability to use different map projections when transforming radar
#   antenna coordinates to Cartesian coordinates including accounting for
#   atmospheric refraction.


def radar_pointing_directions(domain, debug=False, verbose=False):
    """
    Compute radar pointing directions within grid.

    Parameters
    ----------
    domain : Domain
        Domain containing grid coordinates and radar offset from grid origin.

    Optional parameters
    -------------------
    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print relevant information, False to suppress.

    Returns
    -------
    r : ndarray
        Range in meters from radar.
    phi : ndarray
        Radar azimuth angle in degrees measured clockwise from north.
    theta : ndarray
        Radar elevation angle in degrees measured from horizon.

    Notes
    -----

    """

    # Parse grid coordinates and radar offset from grid origin
    za, ya, xa = domain.coordinates
    zr, yr, xr = domain.radar_offset

    # Compute range
    _range = np.sqrt((xa - xr)**2.0 + (ya - yr)**2.0 + (za - zr)**2.0)

    # Compute azimuth direction in degrees clockwise from north
    azimuth = np.degrees(np.arctan2(xa - xr, ya - yr))
    azimuth[azimuth < 0.0] += 360.0

    # Compute elevation direction in degrees from horizon
    elevation = np.degrees(np.arcsin((za - zr) / _range))

    return _range, azimuth, elevation


def equivalent_earth_model(
        radar, offset=None, a=6371.0, ke=4.0/3.0, debug=False, verbose=False):
    """
    Transform radar antenna coordinates to Cartesian coordinates assuming
    standard atmospheric refraction.

    Parameters
    ----------
    radar : pyart.core.Radar
        Radar containing the range, azimuth, and elevation coordinates of
        radar gates.
    offset : sequence of float, optional
        The (z, y, x) offset to apply to radar gate coordinates. If None, no
        offset is applied and returned coordinates are relative to the radar.

    Optional parameters
    -------------------
    a : float, optional
        Earth radius in kilometers. Since the Earth is not a true sphere but
        more of an oblate spheroid, its radius is a function of latitude.
        Earth's polar and equitorial radii are approximately 6357 km and 6378
        km, respectively. Therefore the default value should apply reasonably
        well for radars located at midlatitudes.
    ke : float, optional
        Effective Earth radius multiplier dependent on the vertical gradient of
        the refractive index of air. The default value corresponds to standard
        atmospheric refraction or the 4/3 equivalent Earth radius model.
    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print relevant information, False to suppress.

    Return
    ------
    coordinates : sequence of ndarray
        The (z, y, x) coordinates of radar gates relative to offset.

    Notes
    -----
    The two main assumptions here are that (1) temperature and humidity are
    horizontally homogeneous, i.e., refractive index is a function of height
    only and (2) the Earth is a sphere.

    """

    # Compute effective Earth radius in meters
    ae = ke * a * 1000.0

    # Parse radar longitude, latitude, and altitude above mean sea level
    radar_lon = radar.longitude['data'][0]
    radar_lat = radar.latitude['data'][0]
    radar_alt = radar.altitude['data'][0]

    # Parse radar pointing directions
    # Convert scan angles to radians
    _range = radar.range['data']
    elevation = np.radians(radar.elevation['data'])
    azimuth = np.radians(radar.azimuth['data'])

    # Create radar coordinates mesh
    theta, r = np.meshgrid(elevation, _range, indexing='ij')
    phi, r = np.meshgrid(azimuth, _range, indexing='ij')

    # Compute local elevation correction (delta) due to Earth curvature
    delta = np.arctan(r * np.cos(theta) / (ae + r * np.sin(theta)))

    # Compute arc length or surface range (s), height above surface (h),
    # eastward arc distance along surface (x), and northward arc distance along
    # surface (y)
    h = np.sqrt(r**2.0 + ae**2.0 + 2.0 * r * ae * np.sin(theta)) - ae
    s = ae * np.arcsin(r * np.cos(theta) / (ae + h))
    x = s * np.sin(phi)
    y = s * np.cos(phi)

    # Coordinate transform defined by offset
    if offset is not None:
        z = h + offset[0]
        y += offset[1]
        x += offset[2]

    return z.flatten(), y.flatten(), x.flatten()


def geocentric_radius(lat, geod=None, debug=False, verbose=False):
    """

    Parameters
    ----------
    lat : array_like
        Geodetic latitude in degrees.
    geod : Geod, optional
        A Geod used to define the shape of the Earth. The default datum

    Returns
    -------
    R : ndarray
        The geocentric radius as a function of input latitude using the
        specified geodetic datum.

    """

    # Define geod and parse major (equitorial) and minor (polar) radii
    if geod is None:
        geod = pyproj.Geod(ellps='WGS84', datum='WGS84')
    a, b = geod.a, geod.b

    if debug:
        print 'Major radius (equitorial) : {:.3f} km'.format(a / 1000.0)
        print 'Minor radius (polar) .... : {:.3f} km'.format(b / 1000.0)

    # Parse latitude data and convert degrees to radians
    phi = np.radians(lat)

    # Compute geocentric radius in meters, i.e., the distance from Earth's
    # center to a point on its surface as a function of latitude
    R = np.sqrt(((a**2.0 * np.cos(phi))**2.0 + (b**2.0 * np.sin(phi))**2.0) / \
                ((a * np.cos(phi))**2.0 + (b * np.sin(phi))**2.0))

    return R
