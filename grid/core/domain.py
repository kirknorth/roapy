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
    x, y, z : ndarray
        Axes in meters defining the rectilinear grid.
    coordinates : tuple
        The (z, y, x) coordinates of each vertex in the rectilinear grid.
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

    def __init__(self, axes, origin, proj='lcca', datum='WGS84', ellps='WGS84',
                 dem=None):
        """ Initialize. """

        # Grid axes attributes
        self.z, self.y, self.x = [np.asarray(axis) for axis in axes]
        self.nz, self.ny, self.nx = self.z.size, self.y.size, self.x.size
        self.shape = (self.nz, self.ny, self.nx)

        # Grid origin attributes
        self.lat_0, self.lon_0, self.alt_0 = origin

        # Grid coordinates attribute
        self._add_grid_coordinates()

        # Projection and geod attributes
        self.proj = pyproj.Proj(
            proj=proj, ellps=ellps, datum=datum, lat_0=self.lat_0,
            lon_0=self.lon_0, x_0=0.0, y_0=0.0)
        self.geod = pyproj.Geod(ellps=ellps, datum=datum)

        # GDAL dataset attribute
        self.dem = dem

        # Default radar offset attribute
        self.radar_offset = None


    def compute_radar_offset_from_origin(self, radar, debug=False):
        """ Compute radar (z, y, x) offset from grid origin. """

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
            print 'Radar x offset from origin: {:.2f} m'.format(radar_x)
            print 'Radar y offset from origin: {:.2f} m'.format(radar_y)
            print 'Radar z offset from origin: {:.2f} m'.format(radar_z)

        self.radar_offset = (radar_z, radar_y, radar_x)

        return


    def _add_grid_coordinates(self):
        """ Add (z, y, x) coordinates of each grid point. """
        Z, Y, X = np.meshgrid(self.z, self.y, self.x, indexing='ij')
        self.coordinates = (Z.flatten(), Y.flatten(), X.flatten())
        return

