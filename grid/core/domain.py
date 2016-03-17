"""
grid.core.domain
================

"""

import pyproj
import numpy as np


class Domain(object):
    """

    Attributes
    ----------
    x, y, z : array_like
        Coordinates in meters defining the rectilinear grid.
    lat_0, lon_0, alt_0 : float
        Latitude, longitude, and altitude of the grid origin. Latitude and
        longitude should be in decimal degrees, and altitude should be meters
        above mean sea level.
    proj : str, optional
        Projection used to transform geographic coordinates (latitude and
        longitude) to planar coordinates. The default projection is Lambert
        conformal conic.
    datum : str, optional
        A datum defines the shape, size, and orientation of the Earth. The
        default datum is the World Geodetic System 1984.
    ellps : str, optional
        The default ellipsoid is the World Geodetic System 1984.
    dem : gdal.Dataset, optional
        A digital elevation model (DEM).

    """

    def __init__(self, coords, origin, proj='lcca', datum='WGS84',
                 ellps='WGS84', dem=None):
        """ Initialize. """

        # Grid coordinates and origin attributes
        self.z, self.y, self.x = coords
        self.lat_0, self.lon_0, self.alt_0 = origin
        self.nz, self.ny, self.nx = len(self.z), len(self.y), len(self.x)

        # Projection and geod attributes
        self.proj = pyproj.Proj(
            proj=proj, ellps=ellps, datum=datum, lat_0=self.lat_0,
            lon_0=self.lon_0, x_0=0.0, y_0=0.0)
        self.geod = pyproj.Geod(ellps=ellps, datum=datum)

        # GDAL dataset attribute
        self.dem = dem

        # Radar offset(s) attribute
        self.radar_offset = None


    def radar_offset_from_origin(self, radar, debug=False):
        """ Compute radar (x, y, z) offset from grid origin. """

        # Parse radar latitude, longitude, and altitude
        radar_lat = radar.latitude['data'][0]
        radar_lon = radar.longitude['data'][0]
        radar_alt = radar.altitude['data'][0]

        #
        if self.alt_0 is None:
            self.alt_0 = radar_alt

        #
        radar_x, radar_y = self.proj(radar_lon, radar_lat)
        radar_z = radar_alt - self.alt_0

        if debug:
            print 'Radar x in grid: {:.2f} km'.format(radar_x / 1000.0)
            print 'Radar y in grid: {:.2f} km'.format(radar_y / 1000.0)
            print 'Radar z in grid: {:.2f} km'.format(radar_z / 1000.0)

        self.radar_offset = (radar_z, radar_y, radar_x)

        return

