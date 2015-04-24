"""
grid.interp.mapper
==================

"""

import numpy as np

from scipy import spatial

from pyart.config import get_fillvalue, get_field_name, get_metadata
from pyart.core import Grid

from ..util import transform

# TODO: smartly interpolate the time information of the radar to analysis grid
# points. Note that the time information for a typical radar is the average or
# median time for each ray, i.e., since the sampling of the radar is very quick
# it is overkill to record the time information for each ray and each range
# gate


class Weight(object):
    """
    An object for defining radar data objective analysis parameters.

    Parameters
    ----------

    Attributes
    ----------

    """
    def __init__(self, radar, func=None, cutoff_radius=np.inf, kappa_star=0.5,
                 data_spacing=1220.0):
        """ Initialize. """

        # Default distance-dependent weight parameters
        self.cutoff_radius = cutoff_radius
        self.kappa_star = kappa_star
        self.data_spacing = data_spacing
        self.kappa = None
        self.min_radius = None

        # Distance-dependent weight function and weights
        self.func = func
        self.wq = None

        # The default weighting function is an isotropic Barnes
        # distance-dependent weight with constant smoothing parameter
        if self.func is None:
            self.kappa = kappa_star * (2.0 * data_spacing)**2
            self.func = lambda r: np.ma.exp(-r**2 / self.kappa)

    def compute_weights(self, dist):
        """
        Compute distance-dependent weights.

        Parameters
        ----------
        dist : np.ndarray
            An array of radial distances.
        """
        self.wq = self.func(dist)
        return


def grid_radar(
        radar, grid_coords, weight=None, lat_0=None, lon_0=None, alt_0=None,
        fields=None, toa=17000.0, max_range=None, k=200, leafsize=10, eps=0.0,
        proj='lcc', datum='NAD83', ellps='GRS80', fill_value=None,
        dist_field=None, time_field=None, debug=False, verbose=False):
    """
    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if dist_field is None:
        dist_field = 'nearest_neighbor_distance'
    if time_field is None:
        time_field = 'radar_sampling_time'

    # Parse analysis domain origin
    if lat_0 is None:
        lat_0 = radar.latitude['data'][0]
    if lon_0 is None:
        lon_0 = radar.longitude['data'][0]
    if alt_0 is None:
        alt_0 = radar.altitude['data'][0]

    # Parse fields to map
    if fields is None:
        fields = radar.fields.keys()

    # Parse radar data objective analysis weight
    if weight is None:
        weight = Weight()

    # Parse maximum range
    if max_range is None:
        max_range = radar.range['data'].max()

    # Calculate radar offset relative to the analysis domain origin
    offset = transform._calculate_radar_offset(
        radar, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0, proj=proj, datum=datum,
        ellps=ellps, debug=debug, verbose=verbose)

    # Compute Cartesian coordinates of radar gates and apply origin offset
    z_g, y_g, x_g = transform.standard_refraction(
        radar, debug=debug, verbose=verbose)
    z_g = z_g + offset[0]
    y_g = y_g + offset[1]
    x_g = x_g + offset[2]

    # Do not consider radar gates that are past the "top of the atmosphere"
    # This will speed up processing time during the creation of the k-d tree
    # since it removes unneccessary gates
    is_below_toa = z_g <= toa

    if debug:
        n = is_below_toa.sum()
        print 'Number of radar gates below TOA: {}'.format(n)

    # Slice radar coordinates below the TOA
    z_g = z_g[is_below_toa]
    y_g = y_g[is_below_toa]
    x_g = x_g[is_below_toa]

    # Slice radar fields below the TOA
    # TODO: refactor this section such that the initial radar object is not
    # affected
    for field in fields:
        data = radar.fields[field]['data'].flatten()
        radar.fields[field]['data'] = data[is_below_toa]

    # Create k-d tree object for radar gate locations
    # Depending on the number of radar gates this can be resource intensive
    # but nonetheless should take on the order of 1 second to create
    if debug:
        print 'Creating k-d tree instance for radar gate locations'
    tree_g = spatial.cKDTree(zip(z_g, y_g, x_g), leafsize=leafsize)

    if debug:
        print 'tree.m = {}'.format(tree_g.m)
        print 'tree.n = {}'.format(tree_g.n)

    # Parse Cartesian coordinates of analysis domain
    z_a, y_a, x_a = grid_coords
    nz, ny, nx = z_a.size, y_a.size, x_a.size

    if debug:
        print 'Grid z-size: {}'.format(nz)
        print 'Grid y-size: {}'.format(ny)
        print 'Grid x-size: {}'.format(nx)

    # Create analysis domain coordinates mesh
    z_a, y_a, x_a = np.meshgrid(z_a, y_a, x_a, indexing='ij')
    z_a, y_a, x_a = z_a.flatten(), y_a.flatten(), x_a.flatten()

    if debug:
        print 'Grid 1-D array shape: {}'.format(z_a.shape)

    # Query the radar gate k-d tree for the k-nearest analysis grid points
    # This is the step that consumes the most processing time
    if debug:
        print 'Querying k-d tree for the k-nearest analysis grid points'
    dist, ind = tree_g.query(
        zip(z_a, y_a, x_a), k=k, p=2.0, eps=eps,
        distance_upper_bound=weight.cutoff_radius)

    # Compute distance-dependent weights
    if debug:
        print 'Computing distance-dependent weights'
    weight.compute_weights(dist)

    if debug:
        dist_min, dist_max = dist.min() / 1000.0, dist.max() / 1000.0
        print 'Distance array shape: {}'.format(dist.shape)
        print 'Minimum gate-grid distance: {:.2f} km'.format(dist_min)
        print 'Maximum gate-grid distance: {:.2f} km'.format(dist_max)
        print 'Index array shape: {}'.format(ind.shape)
        print 'Minimum index: {}'.format(ind.min())
        print 'Maximum index: {}'.format(ind.max())

    # Missing neighbors are indicated with an index set to tree.n
    # This condition will not be met for the nearest neighbor scheme, but
    # it can be met for the Cressman and Barnes schemes
    is_bad_index = ind == tree_g.n

    # Analysis grid points which are further than the specified maximum
    # range away from the radar should not be interpolated
    # This is to account for the unambiguous range of the radar, i.e., not all
    # analysis grid points should be considered for interpolation because the
    # radar does not have observations at all distances
    z_r, y_r, x_r = offset
    _range = np.sqrt((z_a - z_r)**2 + (y_a - y_r)**2 + (x_a - x_r)**2)
    is_far = _range > max_range

    if debug:
        n = is_far.sum()
        print 'Number of analysis points too far from radar: {}'.format(n)

    # Interpolate the radar data onto the analysis domain grid
    # Populate the mapped fields dictionary
    map_fields = {}
    for field in fields:
        if debug:
            print 'Mapping radar field: {}'.format(field)

        # Populate field metadata
        map_fields[field] = get_metadata(field)

        # Distance-dependent weighted average defines the interpolation
        # Mask analysis grid points further than the maximum (unambiguous)
        # range
        fq = np.ma.average(
            radar.fields[field]['data'][ind], weights=weight.wq, axis=1)
        fq = np.ma.masked_where(is_far, fq, copy=False)

        # Save interpolated radar field
        map_fields[field]['data'] = fq.reshape(nz, ny, nx).astype(np.float32)

    # Save the nearest neighbor distances
    map_fields[dist_field] = {
        'data': dist.min(axis=1).reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'nearest_neighbor_distance',
        'long_name': 'Nearest neighbor distance',
        'valid_min': 0.0,
        'valid_max': np.inf,
        'units': 'meters',
        '_FillValue': None,
        'comment': '',
        }

    # Interpolate radar time data
    # Analysis grid points are assigned the median time of the k-nearest radar
    # gates
    time = np.repeat(radar.time['data'], radar.ngates).reshape(
        radar.nrays, radar.ngates).flatten()
    fq = np.median(time[is_below_toa][ind], axis=1)
    map_fields[time_field] = {
        'data': fq.reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'radar_sampling_time',
        'long_name': 'Radar sampling time',
        '_FillValue': None,
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        }

    # Create grid axes
    axes = _populate_axes(
        radar, grid_coords, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0)

    # Create grid metadata
    metadata = _populate_metadata(radar)

    return Grid(map_fields, axes, metadata)


def _grid_radar_nearest(
        radar, grid_coords, lat_0=None, lon_0=None, alt_0=None, fields=None,
        toa=17000.0, max_range=None, leafsize=10, eps=0.0, proj='lcc',
        datum='NAD83', ellps='GRS80', fill_value=None, dist_field=None,
        time_field=None, debug=False, verbose=False):
    """
    Map scanning radar data to a Cartesian analysis domain using nearest
    neighbour scheme.

    Parameters
    ----------
    radar : Radar
        A radar object to be mapped to the Cartesian analysis grid.
    grid_coords : tuple or list
        The (z, y, x) coordinates of the grid in meters. These can describe
        either a uniform or non-uniform grid.
    lat_0, lon_0, alt_0 : float
        The latitude, longitude, and altitude AMSL of the grid origin,
        respectively. The default uses the location of the radar as the grid
        origin.
    fields : list
        List of radar fields which will be mapped to the Cartesian analysis
        domain. The default maps all the radar fields.
    toa : float
        The "top of the atmosphere" in meters. Radar gates above this height
        are excluded.
    max_range : float
        The "unambiguous range" of the radar. Analysis grid points which are
        not within this range from the radar are excluded.
    leafsize : int
        Leaf size passed to the cKDTree object. This can affect the processing
        time during the construction and query of the cKDTree, as well as the
        memory required to store the tree. The optimal value depends on the
        nature of the input data. Note that this parameter will not affect
        the results, only the processing time.
    eps : float
        Return approximate nearest neighbors. The k-th returned value is
        guaranteed to be no further than (1 + eps) times the distance to
        the real k-th nearest neighbor.
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
    grid : Grid
        Grid object with mapped radar fields, axes, and metadata.
    """

    # Parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field names
    if dist_field is None:
        dist_field = 'nearest_neighbor_distance'
    if time_field is None:
        time_field = 'radar_sampling_time'

    # Parse analysis domain origin
    if lat_0 is None:
        lat_0 = radar.latitude['data'][0]
    if lon_0 is None:
        lon_0 = radar.longitude['data'][0]
    if alt_0 is None:
        alt_0 = radar.altitude['data'][0]

    # Parse maximum range
    if max_range is None:
        max_range = radar.range['data'].max()

    # Parse fields to map
    if fields is None:
        fields = radar.fields.keys()

    # Calculate radar offset relative to the analysis domain origin
    offset = transform._calculate_radar_offset(
        radar, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0, proj=proj, datum=datum,
        ellps=ellps, debug=debug, verbose=verbose)

    # Compute Cartesian coordinates of radar gates and apply origin offset
    z_g, y_g, x_g = transform.standard_refraction(
        radar, debug=debug, verbose=verbose)
    z_g = z_g + offset[0]
    y_g = y_g + offset[1]
    x_g = x_g + offset[2]

    # Do not consider radar gates that are past the "top of the atmosphere"
    # This will speed up processing time during the creation of the k-d tree
    # since it removes unneccessary gates
    is_below_toa = z_g <= toa

    if debug:
        n = is_below_toa.sum()
        print 'Number of radar gates below TOA: {}'.format(n)

    # Slice radar coordinates below the TOA
    z_g = z_g[is_below_toa]
    y_g = y_g[is_below_toa]
    x_g = x_g[is_below_toa]

    # Slice radar fields below the TOA
    for field in fields:
        data = radar.fields[field]['data'].flatten()
        radar.fields[field]['data'] = data[is_below_toa]

    # Create k-d tree object for radar gate locations
    # Depending on the number of radar gates this can be resource intensive
    # but nonetheless should take on the order of 1 second to create
    if debug:
        print 'Creating k-d tree instance for radar gate locations'
    tree_g = spatial.cKDTree(zip(z_g, y_g, x_g), leafsize=leafsize)

    if debug:
        print 'tree.m = {}'.format(tree_g.m)
        print 'tree.n = {}'.format(tree_g.n)

    # Parse Cartesian coordinates of analysis domain grid
    z_a, y_a, x_a = grid_coords
    nz, ny, nx = z_a.size, y_a.size, x_a.size

    if debug:
        print 'Grid z-size: {}'.format(nz)
        print 'Grid y-size: {}'.format(ny)
        print 'Grid x-size: {}'.format(nx)

    # Create analysis domain coordinates mesh
    z_a, y_a, x_a = np.meshgrid(z_a, y_a, x_a, indexing='ij')
    z_a, y_a, x_a = z_a.flatten(), y_a.flatten(), x_a.flatten()

    if debug:
        print 'Grid 1-D array shape: {}'.format(z_a.shape)

    # Query the radar gate k-d tree for the nearest analysis grid points
    # This is the step that consumes the most processing time
    if debug:
        print 'Querying k-d tree for the k-nearest analysis grid points'
    dist, ind = tree_g.query(
        zip(z_a, y_a, x_a), k=1, p=2.0, eps=eps, distance_upper_bound=np.inf)

    if debug:
        dist_min, dist_max = dist.min() / 1000.0, dist.max() / 1000.0
        print 'Distance array shape: {}'.format(dist.shape)
        print 'Minimum gate-grid distance: {:.2f} km'.format(dist_min)
        print 'Maximum gate-grid distance: {:.2f} km'.format(dist_max)
        print 'Index array shape: {}'.format(ind.shape)
        print 'Minimum index: {}'.format(ind.min())
        print 'Maximum index: {}'.format(ind.max())

    # Analysis grid points which are further than the specified maximum
    # range away from the radar should not be interpolated
    # This is to account for the unambiguous range of the radar, i.e., not all
    # analysis grid points should be considered for interpolation because the
    # radar does not have observations at all distances
    z_r, y_r, x_r = offset
    _range = np.sqrt((z_a - z_r)**2 + (y_a - y_r)**2 + (x_a - x_r)**2)
    is_far = _range > max_range

    if debug:
        n = is_far.sum()
        print 'Number of analysis points too far from radar: {}'.format(n)

    # Interpolate the radar field data onto the analysis domain grid
    # Populate the mapped fields dictionary
    map_fields = {}
    for field in fields:
        if debug:
            print 'Mapping radar field: {}'.format(field)

        # Populate field metadata
        map_fields[field] = get_metadata(field)

        # Nearest neighbours
        # Mask analysis grid points further than the maximum (unambiguous)
        # range
        fq = radar.fields[field]['data'][ind]
        fq = np.ma.masked_where(is_far, fq, copy=False)

        # Save interpolated data
        map_fields[field]['data'] = fq.reshape(nz, ny, nx).astype(np.float32)

    # Save the nearest neighbor distances
    map_fields[dist_field] = {
        'data': dist.reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'nearest_neighbor_distance',
        'long_name': 'Nearest neighbor distance',
        'valid_min': 0.0,
        'valid_max': np.inf,
        'units': 'meters',
        '_FillValue': None,
        'comment': '',
        }

    # Save radar time data
    # Analysis grid points are assigned the time of the closest radar gate
    time = np.repeat(radar.time['data'], radar.ngates).reshape(
        radar.nrays, radar.ngates).flatten()
    fq = time[is_below_toa][ind]
    map_fields[time_field] = {
        'data': fq.reshape(nz, ny, nx).astype(np.float32),
        'standard_name': 'radar_sampling_time',
        'long_name': 'Radar sampling time',
        '_FillValue': None,
        'units': radar.time['units'],
        'calendar': radar.time['calendar'],
        }

    # Create grid axes
    axes = _populate_axes(
        radar, grid_coords, lat_0=lat_0, lon_0=lon_0, alt_0=alt_0)

    # Create grid metadata
    metadata = _populate_metadata(radar)

    return Grid(map_fields, axes, metadata)


def _populate_axes(radar, grid_coords, lat_0=None, lon_0=None, alt_0=0.0):
    """ Populate grid axes """

    # Populate Cartesian axes
    x_disp = {
        'data': grid_coords[2].astype(np.float32),
        'standard_name': 'x_coordinate',
        'long_name': 'x-coordinate in Cartesian system',
        'units': 'meters',
        'axis': 'X',
        'positive': 'east',
        }
    y_disp = {
        'data': grid_coords[1].astype(np.float32),
        'standard_name': 'y_coordinate',
        'long_name': 'y-coordinate in Cartesian system',
        'units': 'meters',
        'axis': 'Y',
        'positive': 'north',
        }
    z_disp = {
        'data': grid_coords[0].astype(np.float32),
        'standard_name': 'z_coordinate',
        'long_name': 'z-coordinate in Cartesian system',
        'units': 'meters',
        'axis': 'Z',
        'positive': 'up',
        }

    # Populate grid origin locations
    altitude = {
        'data': np.array([alt_0], dtype=np.float32),
        'standard_name': 'altitude',
        'long_name': 'Altitude at grid origin above mean sea level',
        'units': 'meters',
        }
    latitude = {
        'data': np.array([lat_0], dtype=np.float32),
        'standard_name': 'latitude',
        'long_name': 'Latitude at grid origin',
        'units': 'degrees_N',
        'valid_min': -90.0,
        'valid_max': 90.0,
        }
    longitude = {
        'data': np.array([lon_0], dtype=np.float32),
        'standard_name': 'longitude',
        'long_name': 'Longitude at grid origin',
        'units': 'degrees_E',
        'valid_min': -180.0,
        'valid_max': 180.0,
        }

    # Populate time information
    time = {
        'data': np.array([radar.time['data'].min()], dtype=np.float64),
        'standard_name': 'time',
        'long_name': 'Time in seconds since radar volume start',
        'units': radar.time['units'],
        'calendar': 'gregorian',
        }
    time_start = {
        'data': radar.time['data'].min().astype(np.float64),
        'standard_name': 'time_start',
        'long_name': 'Time in seconds since radar volume start',
        'units': radar.time['units'],
        'calendar': 'gregorian',
        }
    time_end = {
        'data': radar.time['data'].max().astype(np.float64),
        'standard_name': 'time_end',
        'long_name': 'Time in seconds since radar volume end',
        'units': radar.time['units'],
        'calendar': 'gregorian',
        }
    base_time = {}
    time_offset = {}

    return {
        'time': time,
        'time_start': time_start,
        'time_end': time_end,
        'x_disp': x_disp,
        'y_disp': y_disp,
        'z_disp': z_disp,
        'alt': altitude,
        'lat': latitude,
        'lon': longitude,
        }


def _populate_metadata(radar):
    """
    """
    # Datastreams attribute
    datastream_description = (
        'A string consisting of the datastream(s), datastream version(s), '
        'and datastream date (range).')

    return {
        'process_version': '',
        'references': '',
        'Conventions': '',
        'site_id': '',
        'site': '',
        'facility_id': '',
        'project': '',
        'state': '',
        'comment': '',
        'institution': '',
        'country': '',
        'description': '',
        'title': 'Mapped Moments to Cartesian Grid',
        'project': '',
        'input_datastreams_description': datastream_description,
        'input_datastreams_num': 1,
        'input_datastreams': '',
        'radar_0_altitude': radar.altitude['data'][0],
        'radar_0_latitude': radar.latitude['data'][0],
        'radar_0_longitude': radar.longitude['data'][0],
        'radar_0_instrument_name': radar.metadata['instrument_name'],
        'history': '',
    }
