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


def equivalent_earth_model(
        radar, offset=None, R=6371.0, ke=4.0/3.0, flatten=True, debug=False,
        verbose=False):
    """
    Transform radar antenna coordinates to Cartesian coordinates assuming
    standard atmospheric refraction. The two main assumptions here are that
    temperature and humidity are horizontally homogeneous, i.e., refractive
    index is a function of height only, and the Earth is a sphere.

    Parameters
    ----------
    radar : pyart.core.Radar
        Radar containing the range, azimuth, and elevation coordinates of
        radar gates.
    offset : array_like, optional
        The (z, y, x) offset to apply to radar gate coordinates. If offset is
        None then no offset is applied, i.e., Cartesian coordinates are
        relative to the radar.
    R : float, optional
        Earth radius in kilometers. Since the Earth is not a true sphere but
        more of an oblate spheroid, its radius is a function of latitude.
        Earth's polar radius is approximately 6357 km while its equitorial
        radius is close to 6378 km. Therefore the default value should apply
        reasonably well for radars located at midlatitudes.
    ke : float, optional
        Effective Earth radius multiplier dependent on the vertical gradient of
        the refractive index of air. The default value corresponds to standard
        atmospheric refraction or the 4/3 equivalent Earth radius model.
    flatten : bool, optional
        True to flatten radar gate coordinates, False to return original
        dimensions. If the user wishes to return the (x, y, z) locations of the
        range gates in the original order then set to False.
    debug : bool, optional
        True to print debugging information, False to suppress.
    verbose : bool, optional
        True to print relevant information, False to suppress.

    Return
    ------
    coords : array_like
        The (z, y, x) coordinates of radar gates relative to the offset.

    """

    # Equivalent Earth radius in meters
    Re = R * ke * 1000.0

    # Parse radar longitude, latitude, and altitude above mean sea level
    radar_lon = radar.longitude['data'][0]
    radar_lat = radar.latitude['data'][0]
    radar_alt = radar.altitude['data'][0]

    # Parse radar pointing directions
    # Convert scan angles to radians
    rng = radar.range['data']
    ele = np.radians(radar.elevation['data'])
    azi = np.radians(radar.azimuth['data'])

    # Create radar coordinates mesh
    ELE, RNG = np.meshgrid(ele, rng, indexing='ij')
    AZI, RNG = np.meshgrid(azi, rng, indexing='ij')
    if flatten:
        RNG, AZI, ELE = RNG.flatten(), AZI.flatten(), ELE.flatten()

    # Compute vertical height (z), arc length (s), eastward distance (x),
    # and northward distance (y)
    # These equations contain an implicit map projection assumption, most
    # likely the azimuthal equidistant projection
    z = np.sqrt(RNG**2 + 2.0 * RNG * Re * np.sin(ELE) + Re**2) - Re
    s = Re * np.arcsin(RNG * np.cos(ELE) / (z + Re))
    x = s * np.sin(AZI)
    y = s * np.cos(AZI)

    # Coordinate transform defined by offset
    if offset is not None:
        z += offset[0]
        y += offset[1]
        x += offset[2]

    return z, y, x


def geocentric_radius(lat, geod=None, verbose=False):
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

    if verbose:
        print 'Major radius (equitorial) : {:.3f} km'.format(a / 1000.0)
        print 'Minor radius (polar) .... : {:.3f} km'.format(b / 1000.0)

    # Parse latitude data and convert degrees to radians
    phi = np.radians(lat)

    # Compute geocentric radius in meters, i.e., the distance from Earth's
    # center to a point on its surface as a function of latitude
    R = np.sqrt(((a**2 * np.cos(phi))**2 + (b**2 * np.sin(phi))**2) / \
                ((a * np.cos(phi))**2 + (b * np.sin(phi))**2))

    return R
